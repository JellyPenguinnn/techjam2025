import os
import json
import argparse
import ast
from typing import List, Dict, Tuple, Optional

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer, EarlyStoppingCallback, set_seed
from transformers import DebertaV2TokenizerFast
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
import inspect


def parse_args():
    p = argparse.ArgumentParser()
    # Model setup
    p.add_argument('--model', default='microsoft/deberta-v3-small', help='Model to fine-tune (HuggingFace ID or local path)')
    p.add_argument('--output_dir', default='./models/deberta-ner-finetuned', help='Where to save the trained model')
    p.add_argument('--max_length', type=int, default=128, help='Max text length for training')

    # Training data
    p.add_argument('--pii_csv', default='pii_dataset.csv', help='Main PII training data (CSV with tokens and labels columns)')
    p.add_argument('--tokens_col', default='tokens', help='CSV column with word tokens')
    p.add_argument('--labels_col', default='labels', help='CSV column with BIO labels')
    p.add_argument('--conll2003_dir', default='.', help='CoNLL-2003 dataset folder (train.txt, valid.txt, test.txt)')
    p.add_argument('--pseudolabel_csv', default=None, help='Additional pseudo-labeled data (optional)')
    p.add_argument('--pseudolabel_txt', default='pseudo_input.txt', help='Raw text to auto-label (one doc per paragraph)')
    p.add_argument('--pseudolabel_model', default='dslim/bert-base-NER', help='Model for generating pseudo-labels')
    p.add_argument('--pseudo_limit', type=int, default=None, help='Max number of pseudo-labeled documents')
    p.add_argument('--pseudo_batch_size', type=int, default=4, help='Batch size for pseudo-labeling')
    p.add_argument('--pseudo_token_limit', type=int, default=None, help='Max pseudo-labeled tokens to use')
    # Data balancing
    p.add_argument('--disable_pseudo', action='store_true', help='Skip pseudo-labeled data completely')
    p.add_argument('--disable_conll', action='store_true', help='Skip CoNLL-2003 data completely')
    p.add_argument('--pseudo_max_ratio', type=float, default=None, help='Limit pseudo data vs PII data ratio (e.g., 0.5 = max 50%)')
    p.add_argument('--conll_max_ratio', type=float, default=None, help='Limit CoNLL data vs PII data ratio')
    p.add_argument('--pseudo_label_filter', default='', help='Entity types to keep from pseudo labels (e.g., PER,ORG)')

    # Training strategy
    p.add_argument('--val_split', type=float, default=0.1, help='Validation split (0.1 = 10%)')
    p.add_argument('--test_split', type=float, default=0.1, help='Test split (0.1 = 10%)')
    p.add_argument('--stratify', action='store_true', help='Balanced splits based on entity presence')
    p.add_argument('--training_strategy', default='joint', choices=['joint', 'sequential'], help='Train on all data together (joint) or in stages (sequential)')
    p.add_argument('--pii_oversample_factor', type=float, default=2.0, help='How much to oversample PII data (2.0 = double)')
    p.add_argument('--augment', action='store_true', help='Apply data augmentation (case changes)')
    p.add_argument('--curriculum_epochs', type=int, default=0, help='Start with easier data, add pseudo-labels after N epochs')

    # Training settings
    p.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    p.add_argument('--epochs_stage1', type=int, default=None, help='Epochs for CoNLL training (sequential mode)')
    p.add_argument('--epochs_stage2', type=int, default=None, help='Epochs for PII training (sequential mode)')
    p.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    p.add_argument('--batch', type=int, default=4, help='Batch size per GPU')
    p.add_argument('--grad_accum', type=int, default=1, help='Gradient accumulation steps')
    p.add_argument('--warmup_ratio', type=float, default=0.1, help='Learning rate warmup (0.1 = 10% of training)')
    p.add_argument('--label_smoothing', type=float, default=0.05, help='Label smoothing factor')
    p.add_argument('--early_stopping_patience', type=int, default=2, help='Stop if no improvement for N evaluations (0 = disabled)')
    p.add_argument('--gradient_checkpointing', action='store_true', help='Save memory by recomputing gradients')
    p.add_argument('--weight_decay', type=float, default=0.01, help='L2 regularization strength')
    p.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping (0 = disabled)')
    p.add_argument('--dataloader_num_workers', type=int, default=2, help='Data loading threads')
    p.add_argument('--precision', default='auto', choices=['auto','fp32','fp16','bf16'], help='Training precision')
    p.add_argument('--resume_from_checkpoint', default=None, help='Resume from checkpoint directory')
    p.add_argument('--use_mps', action='store_true', help='Use Apple Metal GPU (experimental)')
    p.add_argument('--force_cpu', action='store_true', help='Force CPU training')
    p.add_argument('--map_batch_size', type=int, default=512, help='Batch size for data preprocessing')
    p.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    p.add_argument('--report_to', default='none', choices=['none','wandb','tensorboard'], help='Logging service')
    p.add_argument('--auto_find_batch_size', action='store_true', help='Auto-reduce batch size on GPU memory errors')

    # Model export
    p.add_argument('--export_int8', action='store_true', help='Create int8 quantized model for faster inference')
    p.add_argument('--export_onnx', action='store_true', help='Export ONNX format for cross-platform use')
    p.add_argument('--onnx_opset', type=int, default=13, help='ONNX version compatibility')
    return p.parse_args()


def _sanitize_bio(labels: List[str]) -> List[str]:
    """Fix broken BIO label sequences (converts invalid I- tags to B-)"""
    fixed: List[str] = []
    prev_type: str = ''
    prev_prefix: str = 'O'
    for tag in labels:
        if tag == 'O':
            fixed.append(tag)
            prev_type, prev_prefix = '', 'O'
            continue
        if '-' in tag:
            prefix, t = tag.split('-', 1)
        else:
            prefix, t = 'B', tag
        if prefix == 'I' and not (prev_prefix in ('B', 'I') and prev_type == t):
            prefix = 'B'
        fixed.append(f"{prefix}-{t}")
        prev_type, prev_prefix = t, prefix
    return fixed


def _order_bio_labels(label_set: set) -> List[str]:
    def sort_key(tag: str):
        if tag == 'O':
            return (0, '', 0)
        if '-' in tag:
            prefix, t = tag.split('-', 1)
        else:
            prefix, t = 'B', tag
        prefix_rank = 0 if prefix == 'B' else 1
        return (1, t, prefix_rank)
    return sorted(label_set, key=sort_key)


def _pretokenize_words(text: str, tok) -> Tuple[List[str], List[int]]:
    """Split text into words and track where each word starts"""
    pre = getattr(tok.backend_tokenizer, 'pre_tokenizer', None)
    if pre is None:
        raise RuntimeError('Tokenizer pre_tokenizer is not available; cannot safely align pseudo-labels. Use a fast tokenizer.')
    parts = pre.pre_tokenize_str(text)
    tokens = [w for (w, (s, e)) in parts]
    starts = [s for (w, (s, e)) in parts]
    return tokens, starts


def _char_spans_to_bio_with_tokenizer(text: str, spans: List[Tuple[int, int, str]], tok) -> Tuple[List[str], List[str]]:
    """Convert character-level labels to word-level BIO tags"""
    tokens, starts = _pretokenize_words(text, tok)
    labels = ['O'] * len(tokens)
    for (s, e, lab) in spans:
        i = 0
        started = False
        while i < len(tokens):
            ts = starts[i]
            te = ts + len(tokens[i])
            if te <= s:
                i += 1
                continue
            if ts >= e:
                break
            labels[i] = f"B-{lab}" if not started else f"I-{lab}"
            started = True
            i += 1
    return tokens, _sanitize_bio(labels)


def load_pii_csv_with_splits(csv_path: str, tokens_col: str, labels_col: str, val_split: float, test_split: float, stratify: bool, seed: int) -> Dict[str, Tuple[List[List[str]], List[List[str]]]]:
    """Load PII data from CSV and split into train/validation/test sets"""
    df = pd.read_csv(csv_path)
    if tokens_col not in df.columns or labels_col not in df.columns:
        raise ValueError(f'CSV must include columns {tokens_col} and {labels_col}')
    tokens_list: List[List[str]] = []
    labels_list: List[List[str]] = []
    for _, row in df.iterrows():
        try:
            toks = ast.literal_eval(row[tokens_col])
            labs = ast.literal_eval(row[labels_col])
            if not isinstance(toks, list) or not isinstance(labs, list):
                continue
            if len(toks) != len(labs):
                continue
            tokens_list.append([str(t) for t in toks])
            labels_list.append(_sanitize_bio([str(l) for l in labs]))
        except Exception:
            continue

    sdf = pd.DataFrame({
        'tokens': tokens_list,
        'labels': labels_list,
        'has_pii': [any(l != 'O' for l in labs) for labs in labels_list],
    })
    # Compute sizes
    val_ratio = max(0.0, min(1.0, val_split))
    test_ratio = max(0.0, min(1.0, test_split))
    if val_ratio + test_ratio >= 1.0:
        raise ValueError('val_split + test_split must be < 1.0')

    if stratify and sdf['has_pii'].nunique() > 1:
        temp_df, test_df = train_test_split(sdf, test_size=test_ratio, random_state=seed, stratify=sdf['has_pii']) if test_ratio > 0 else (sdf, None)
        if val_ratio > 0:
            temp_has = temp_df['has_pii'] if test_ratio > 0 else sdf['has_pii']
            train_df, val_df = train_test_split(temp_df, test_size=val_ratio/(1.0 - test_ratio), random_state=seed, stratify=temp_has)
        else:
            train_df, val_df = temp_df, None
    else:
        temp_df, test_df = train_test_split(sdf, test_size=test_ratio, random_state=seed) if test_ratio > 0 else (sdf, None)
        if val_ratio > 0:
            train_df, val_df = train_test_split(temp_df, test_size=val_ratio/(1.0 - test_ratio), random_state=seed)
        else:
            train_df, val_df = temp_df, None

    splits: Dict[str, Tuple[List[List[str]], List[List[str]]]] = {}
    splits['train'] = (list(train_df['tokens']), list(train_df['labels']))
    if val_df is not None:
        splits['validation'] = (list(val_df['tokens']), list(val_df['labels']))
    if test_df is not None:
        splits['test'] = (list(test_df['tokens']), list(test_df['labels']))
    return splits


def load_local_conll2003(conll_dir: str) -> Dict[str, Tuple[List[List[str]], List[List[str]]]]:
    """Load CoNLL-2003 dataset files for general entity recognition training"""
    def read_conll(path: str) -> Tuple[List[List[str]], List[List[str]]]:
        sents_toks: List[List[str]] = []
        sents_labs: List[List[str]] = []
        cur_toks: List[str] = []
        cur_labs: List[str] = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line:
                    if cur_toks:
                        sents_toks.append(cur_toks)
                        sents_labs.append(cur_labs)
                        cur_toks, cur_labs = [], []
                    continue
                if 'DOCSTART' in line.upper():
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                tok = parts[0]
                ner = parts[-1]
                sents_toks.append([tok]) if False else None  # placate linters if needed
                cur_toks.append(tok)
                cur_labs.append(ner)
        if cur_toks:
            sents_toks.append(cur_toks)
            sents_labs.append(cur_labs)
        return sents_toks, sents_labs

    train_p = os.path.join(conll_dir, 'train.txt')
    valid_p = os.path.join(conll_dir, 'valid.txt')
    test_p = os.path.join(conll_dir, 'test.txt')
    splits: Dict[str, Tuple[List[List[str]], List[List[str]]]] = {}
    if os.path.exists(train_p):
        splits['train'] = read_conll(train_p)
    if os.path.exists(valid_p):
        splits['validation'] = read_conll(valid_p)
    if os.path.exists(test_p):
        splits['test'] = read_conll(test_p)
    return splits


def generate_pseudolabels_from_txt(text_path: str, model_id: str, training_tokenizer, limit: Optional[int] = None, batch_size: int = 4, use_mps: bool = False, token_limit: Optional[int] = None, chunk_words: int = 128, allowed_groups: Optional[set] = None) -> Tuple[List[List[str]], List[List[str]]]:
    """Auto-label raw text using an existing NER model"""
    from transformers import AutoTokenizer as _ATok, AutoModelForTokenClassification as _AModel, pipeline as _pipeline
    if not os.path.exists(text_path):
        return [], []
    with open(text_path, 'r', encoding='utf-8') as f:
        raw = f.read()
    # break text into paragraphs, then into chunks
    paras: List[str] = [doc.strip() for doc in raw.split("\n\n") if doc.strip()]
    docs: List[str] = []
    for para in paras:
        words = para.split()
        if not words:
            continue
        start = 0
        while start < len(words):
            end = min(len(words), start + max(1, chunk_words))
            docs.append(" ".join(words[start:end]))
            start = end

    tok = _ATok.from_pretrained(model_id, use_fast=True)
    mdl = _AModel.from_pretrained(model_id)
    # try to use the best available device
    nlp = None
    if use_mps and torch.backends.mps.is_available():
        try:
            nlp = _pipeline('token-classification', model=mdl, tokenizer=tok, aggregation_strategy='simple', device='mps')
            print('Pseudo-label pipeline device: mps')
        except Exception:
            nlp = None
            print('Pseudo-label pipeline device: mps failed; falling back')
    if nlp is None:
        if torch.cuda.is_available():
            try:
                nlp = _pipeline('token-classification', model=mdl, tokenizer=tok, aggregation_strategy='simple', device=0)
                print('Pseudo-label pipeline device: cuda:0')
            except Exception:
                nlp = _pipeline('token-classification', model=mdl, tokenizer=tok, aggregation_strategy='simple', device=-1)
                print('Pseudo-label pipeline device: cpu (fallback)')
        else:
            nlp = _pipeline('token-classification', model=mdl, tokenizer=tok, aggregation_strategy='simple', device=-1)
            print('Pseudo-label pipeline device: cpu')

    pl_tokens: List[List[str]] = []
    pl_labels: List[List[str]] = []
    count = 0
    total_tokens = 0
    empty_ents = 0

    bs = max(1, int(batch_size) if batch_size is not None else 4)
    stop = False
    with torch.inference_mode():
        for start_idx in range(0, len(docs), bs):
            if stop:
                break
            batch_docs = docs[start_idx:start_idx + bs]
            if not batch_docs:
                continue
            batch_outputs = nlp(batch_docs)
            for doc, ents in zip(batch_docs, batch_outputs):
                spans: List[Tuple[int, int, str]] = []
                for e in ents:
                    s = int(e.get('start', 0))
                    ed = int(e.get('end', 0))
                    lab = str(e.get('entity_group', 'O'))
                    lab = lab.replace('MISC', 'MISC').replace('PER', 'PER').replace('ORG', 'ORG').replace('LOC', 'LOC')
                    if allowed_groups and lab not in allowed_groups:
                        continue
                    spans.append((s, ed, lab))
                if not spans:
                    empty_ents += 1
                tks, labs = _char_spans_to_bio_with_tokenizer(doc, spans, training_tokenizer)
                if tks and labs and len(tks) == len(labs):
                    pl_tokens.append(tks)
                    pl_labels.append(labs)
                    count += 1
                    total_tokens += len(tks)
                    if (limit and count >= limit) or (token_limit and total_tokens >= token_limit):
                        stop = True
                        break
    print(f"Pseudo-labels: processed_docs={len(docs)}, emitted_rows={len(pl_tokens)}, empty_entities={empty_ents}, total_tokens={total_tokens}")
    return pl_tokens, pl_labels


def _augment_tokens(tokens: List[str]) -> List[str]:
    augmented: List[str] = []
    for tok in tokens:
        if any(ch.isalpha() for ch in tok):
            augmented.append(tok.swapcase())
        else:
            augmented.append(tok)
    return augmented if len(augmented) == len(tokens) else tokens


def _apply_augmentation(tokens: List[List[str]], labels: List[List[int]], flags: List[bool], label2id: Dict[str, int], enabled: bool) -> Tuple[List[List[str]], List[List[int]], List[bool]]:
    if not enabled:
        return tokens, labels, flags
    aug_toks: List[List[str]] = []
    aug_labs: List[List[int]] = []
    aug_flags: List[bool] = []
    for tks, labs, is_pii in zip(tokens, labels, flags):
        if is_pii and any(l != label2id['O'] for l in labs):
            at = _augment_tokens(list(tks))
            if len(at) == len(labs):
                aug_toks.append(at)
                aug_labs.append(labs)
                aug_flags.append(is_pii)
    if aug_toks:
        tokens = tokens + aug_toks
        labels = labels + aug_labs
        flags = flags + aug_flags
    return tokens, labels, flags


def align_labels_with_tokens(labels: List[int], word_ids, label_all_tokens: bool = False) -> List[int]:
    """Match word labels to subword pieces from tokenizer"""
    new_labels: List[int] = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            new_labels.append(-100)
        elif word_idx != previous_word_idx:
            new_labels.append(labels[word_idx])
        else:
            new_labels.append(labels[word_idx] if label_all_tokens else -100)
        previous_word_idx = word_idx
    return new_labels


def build_union_label_map(all_label_str_lists: List[List[str]]) -> Tuple[List[str], Dict[str, int]]:
    """Create master label list from all datasets"""
    label_set = set(l for labs in all_label_str_lists for l in labs)
    if 'O' not in label_set:
        label_set.add('O')
    label_list = _order_bio_labels(label_set)
    return label_list, {l: i for i, l in enumerate(label_list)}


def encode_labels(labels_str: List[List[str]], label2id: Dict[str, int]) -> List[List[int]]:
    return [[label2id[l] for l in labs] for labs in labels_str]


def make_hf_dataset(tokens: List[List[str]], labels: List[List[int]], is_pii_flags: Optional[List[bool]] = None) -> Dataset:
    data = {'tokens': tokens, 'ner_tags': labels}
    if is_pii_flags is not None:
        data['is_pii'] = is_pii_flags
    return Dataset.from_dict(data)


def main():
    args = parse_args()

    # set up tokenizer
    if 'deberta' in args.model.lower():
        tokenizer = DebertaV2TokenizerFast.from_pretrained(args.model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # reproducible training
    set_seed(args.seed)
    # prevent multiprocessing issues
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    # load training data
    pii_splits = load_pii_csv_with_splits(args.pii_csv, args.tokens_col, args.labels_col, args.val_split, args.test_split, args.stratify, args.seed)
    conll_splits = {} if args.disable_conll else load_local_conll2003(args.conll2003_dir)

    # show dataset info
    def _len2(split: Tuple[List[List[str]], List[List[str]]]) -> int:
        return len(split[0]) if split else 0
    print(f"PII sizes: train={_len2(pii_splits.get('train'))}, val={_len2(pii_splits.get('validation'))}, test={_len2(pii_splits.get('test'))}")
    print(f"CoNLL sizes: train={_len2(conll_splits.get('train'))}, val={_len2(conll_splits.get('validation'))}, test={_len2(conll_splits.get('test'))}")

    # get additional training data from auto-labeling
    pseudo_tokens: List[List[str]] = []
    pseudo_labels: List[List[str]] = []
    if not args.disable_pseudo:
        if args.pseudolabel_csv and os.path.exists(args.pseudolabel_csv):
            # read pre-labeled CSV
            df = pd.read_csv(args.pseudolabel_csv)
            if args.tokens_col in df.columns and args.labels_col in df.columns:
                for _, row in df.iterrows():
                    try:
                        toks = ast.literal_eval(row[args.tokens_col])
                        labs = ast.literal_eval(row[args.labels_col])
                        if isinstance(toks, list) and isinstance(labs, list) and len(toks) == len(labs):
                            pseudo_tokens.append([str(t) for t in toks])
                            pseudo_labels.append(_sanitize_bio([str(l) for l in labs]))
                    except Exception:
                        continue
        elif args.pseudolabel_txt and os.path.exists(args.pseudolabel_txt):
            allowed = set([s.strip() for s in args.pseudo_label_filter.split(',') if s.strip()]) if args.pseudo_label_filter else None
            pt, pl = generate_pseudolabels_from_txt(
                args.pseudolabel_txt,
                args.pseudolabel_model,
                tokenizer,
                args.pseudo_limit,
                args.pseudo_batch_size,
                use_mps=args.use_mps,
                token_limit=args.pseudo_token_limit,
                allowed_groups=allowed,
            )
            pseudo_tokens, pseudo_labels = pt, pl
            print(f"Generated {len(pseudo_tokens)} pseudo-labeled docs from {args.pseudolabel_txt}.")

    # create unified label vocabulary
    all_label_strings: List[List[str]] = []
    for split in pii_splits.values():
        all_label_strings.extend(split[1])
    for split in conll_splits.values():
        all_label_strings.extend(split[1])
    all_label_strings.extend(pseudo_labels)
    label_list, label2id = build_union_label_map(all_label_strings)
    id2label = {i: l for i, l in enumerate(label_list)}

    # save training configuration
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        with open(os.path.join(args.output_dir, 'run_config.json'), 'w') as f:
            json.dump({k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in vars(args).items()}, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to save run_config.json: {e}")

    # convert text labels to numbers
    pii_enc = {k: (v[0], encode_labels(v[1], label2id)) for k, v in pii_splits.items()}
    conll_enc = {k: (v[0], encode_labels(v[1], label2id)) for k, v in conll_splits.items()}
    pseudo_enc_labels = encode_labels(pseudo_labels, label2id) if pseudo_labels else []

    # limit dataset size based on ratio
    def cap_by_ratio(tokens: List[List[str]], labels: List[List[int]], max_ratio: Optional[float], base_count: int) -> Tuple[List[List[str]], List[List[int]]]:
        if max_ratio is None or max_ratio <= 0 or base_count <= 0:
            return tokens, labels
        cap = max(0, int(base_count * max_ratio))
        if cap and len(tokens) > cap:
            return tokens[:cap], labels[:cap]
        return tokens, labels

    # set up training datasets
    metrics_report: Dict[str, Dict[str, float]] = {}

    def tokenize_and_align_dataset(ds: Dataset) -> Dataset:
        def tokenize_and_align(examples):
            toks = examples['tokens']
            tokenized = tokenizer(toks, truncation=True, max_length=args.max_length, is_split_into_words=True)
            labels = []
            for i, label in enumerate(examples['ner_tags']):
                word_ids = tokenized.word_ids(batch_index=i)
                labels.append(align_labels_with_tokens(label, word_ids))
            tokenized['labels'] = labels
            return tokenized

        return ds.map(tokenize_and_align, batched=True, batch_size=args.map_batch_size)

    # prepare test datasets
    eval_sets: Dict[str, Dataset] = {}
    if 'validation' in pii_enc:
        eval_sets['pii_val'] = make_hf_dataset(pii_enc['validation'][0], pii_enc['validation'][1], [True] * len(pii_enc['validation'][0]))
    if 'test' in pii_enc:
        eval_sets['pii_test'] = make_hf_dataset(pii_enc['test'][0], pii_enc['test'][1], [True] * len(pii_enc['test'][0]))
    if 'validation' in conll_enc:
        eval_sets['conll_valid'] = make_hf_dataset(conll_enc['validation'][0], conll_enc['validation'][1], [False] * len(conll_enc['validation'][0]))
    if 'test' in conll_enc:
        eval_sets['conll_test'] = make_hf_dataset(conll_enc['test'][0], conll_enc['test'][1], [False] * len(conll_enc['test'][0]))

    # Metric function
    def compute_metrics(p):
        if hasattr(p, 'predictions') and hasattr(p, 'label_ids'):
            preds, labels = p.predictions, p.label_ids
        else:
            preds, labels = p
        preds = np.argmax(preds, axis=2)
        true_preds = [[id2label[p] for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(preds, labels)]
        true_labels = [[id2label[l] for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(preds, labels)]
        overall = {
            'precision': precision_score(true_labels, true_preds),
            'recall': recall_score(true_labels, true_preds),
            'f1': f1_score(true_labels, true_preds),
        }
        try:
            from seqeval.metrics import classification_report
            report = classification_report(true_labels, true_preds, output_dict=True)
            per_entity: Dict[str, Dict[str, float]] = {}
            for k, v in report.items():
                if isinstance(v, dict) and k not in ('micro avg','macro avg','weighted avg','accuracy'):
                    per_entity[k] = {
                        'precision': v.get('precision'),
                        'recall': v.get('recall'),
                        'f1': v.get('f1-score'),
                    }
            overall['per_entity'] = per_entity
            overall['macro_f1'] = report.get('macro avg', {}).get('f1-score')
        except Exception:
            print('Warning: seqeval classification_report failed; returning overall metrics only.')
        return overall

    def make_trainer(model, train_ds: Dataset, eval_ds: Optional[Dataset], num_train_epochs: int) -> Trainer:
        per_epoch_steps = max(1, int(np.ceil(len(train_ds) / max(1, (args.batch * args.grad_accum)))))
        logging_steps = max(1, per_epoch_steps // 10)
        # Precision policy
        use_fp16 = False
        use_bf16 = False
        cuda_available = torch.cuda.is_available() and not args.force_cpu
        if args.precision == 'auto':
            use_bf16 = cuda_available and torch.cuda.is_bf16_supported()
            use_fp16 = cuda_available and not use_bf16
        elif args.precision == 'fp16':
            use_fp16 = cuda_available
        elif args.precision == 'bf16':
            use_bf16 = cuda_available and torch.cuda.is_bf16_supported()
        # Build base kwargs safely across transformers versions
        sig = inspect.signature(TrainingArguments.__init__)
        def supports(param_name: str) -> bool:
            return param_name in sig.parameters
        base_kwargs = dict(
            output_dir=args.output_dir,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            num_train_epochs=num_train_epochs,
            weight_decay=args.weight_decay,
            logging_steps=logging_steps,
            save_total_limit=2,
            warmup_ratio=args.warmup_ratio,
            label_smoothing_factor=args.label_smoothing,
            fp16=use_fp16,
            bf16=use_bf16,
            seed=args.seed,
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_accumulation_steps=args.grad_accum,
            dataloader_pin_memory=False,
            dataloader_num_workers=args.dataloader_num_workers,
            max_grad_norm=args.max_grad_norm,
            report_to=[] if args.report_to == 'none' else [args.report_to],
        )
        if supports('use_mps_device'):
            base_kwargs['use_mps_device'] = (args.use_mps and not args.force_cpu)
        # Auto batch size scaling: enable on MPS by default or if user requested
        if supports('auto_find_batch_size') and (args.auto_find_batch_size or (args.use_mps and not args.force_cpu)):
            base_kwargs['auto_find_batch_size'] = True
        # Enable eval if we have an eval dataset
        if supports('do_eval') and eval_ds is not None:
            base_kwargs['do_eval'] = True
        # Eval/save strategy and best-model settings (use eval_strategy in this version)
        supports_eval = supports('eval_strategy')
        supports_save = supports('save_strategy')
        supports_metric = supports('metric_for_best_model') and supports('greater_is_better')
        supports_load_best = supports('load_best_model_at_end')
        if supports_eval and supports_save:
            base_kwargs['eval_strategy'] = 'epoch'
            base_kwargs['save_strategy'] = 'epoch'
            if supports('logging_strategy'):
                base_kwargs['logging_strategy'] = 'epoch'
            if supports_metric:
                base_kwargs['metric_for_best_model'] = 'f1'
                base_kwargs['greater_is_better'] = True
            if supports_load_best:
                base_kwargs['load_best_model_at_end'] = True
        if supports('gradient_checkpointing_kwargs') and args.gradient_checkpointing:
            base_kwargs['gradient_checkpointing_kwargs'] = {'use_reentrant': False}
        training_args = TrainingArguments(**base_kwargs)
        data_collator = DataCollatorForTokenClassification(tokenizer)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds if (eval_ds is not None) else (train_ds.select(range(min(32, len(train_ds)))) if len(train_ds) > 0 else None),
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        if args.early_stopping_patience and args.early_stopping_patience > 0 and supports_eval and supports_save and supports_metric and supports_load_best:
            trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
        return trainer

    # Build model with unified head
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    # Strategy execution
    final_trainer = None
    if args.training_strategy == 'sequential':
        # Stage 1: CoNLL (train), eval on CoNLL valid
        if args.disable_conll or 'train' not in conll_enc or 'validation' not in conll_enc:
            raise RuntimeError('Sequential strategy requires CoNLL train/validation splits unless --disable_conll is set. Remove sequential or provide valid.txt.')
        conll_train_tokens, conll_train_labels = conll_enc['train']
        # Optional cap CoNLL in stage 1 by ratio vs PII count
        base_pii_count = len(pii_enc['train'][0]) if 'train' in pii_enc else 0
        conll_train_tokens, conll_train_labels = cap_by_ratio(conll_train_tokens, conll_train_labels, args.conll_max_ratio, base_pii_count)
        conll_train = make_hf_dataset(conll_train_tokens, conll_train_labels, [False] * len(conll_train_tokens))
        conll_val = eval_sets.get('conll_valid')
        tok_conll_train = tokenize_and_align_dataset(conll_train)
        tok_conll_val = tokenize_and_align_dataset(conll_val)

        e1 = args.epochs_stage1 if args.epochs_stage1 is not None else max(1, args.epochs // 2)
        trainer1 = make_trainer(model, tok_conll_train, tok_conll_val, e1)
        trainer1.train(resume_from_checkpoint=args.resume_from_checkpoint)
        model = trainer1.model  # best model loaded

        # Stage 2: PII train (+ pseudo, oversample), eval on PII val
        if 'train' not in pii_enc or 'validation' not in pii_enc:
            raise RuntimeError('Sequential strategy requires PII train/validation splits in CSV.')
        pii_tokens_train, pii_labels_train = pii_enc['train']
        is_pii_flags = [True] * len(pii_tokens_train)
        # Add pseudo (train only), with ratio cap
        if not args.disable_pseudo and pseudo_tokens:
            base_count = len(pii_tokens_train)
            pt, pl = pseudo_tokens, pseudo_enc_labels
            pt, pl = cap_by_ratio(pt, pl, args.pseudo_max_ratio, base_count)
            pii_tokens_train = pii_tokens_train + pt
            pii_labels_train = pii_labels_train + pl
            is_pii_flags += [True] * len(pt)
        # Oversample PII
        if args.pii_oversample_factor and args.pii_oversample_factor > 1.0:
            extra_tokens = []
            extra_labels = []
            extra_flags = []
            reps = int(args.pii_oversample_factor) - 1
            frac = args.pii_oversample_factor - int(args.pii_oversample_factor)
            base_tokens = list(pii_tokens_train)
            base_labels = list(pii_labels_train)
            base_flags = list(is_pii_flags)
            for _ in range(max(0, reps)):
                extra_tokens.extend(base_tokens)
                extra_labels.extend(base_labels)
                extra_flags.extend(base_flags)
            if frac > 0 and base_tokens:
                take = int(len(base_tokens) * frac)
                extra_tokens.extend(base_tokens[:take])
                extra_labels.extend(base_labels[:take])
                extra_flags.extend(base_flags[:take])
            pii_tokens_train += extra_tokens
            pii_labels_train += extra_labels
            is_pii_flags += extra_flags

        # Apply augmentation before building dataset
        pii_tokens_train, pii_labels_train, is_pii_flags = _apply_augmentation(
            pii_tokens_train, pii_labels_train, is_pii_flags, label2id, args.augment
        )
        pii_train_ds = make_hf_dataset(pii_tokens_train, pii_labels_train, is_pii_flags)
        pii_val_ds = eval_sets['pii_val']
        tok_pii_train = tokenize_and_align_dataset(pii_train_ds)
        tok_pii_val = tokenize_and_align_dataset(pii_val_ds)

        e2 = args.epochs_stage2 if args.epochs_stage2 is not None else max(1, args.epochs - e1)
        trainer2 = make_trainer(model, tok_pii_train, tok_pii_val, e2)
        trainer2.train(resume_from_checkpoint=args.resume_from_checkpoint)
        model = trainer2.model
        final_trainer = trainer2

        # Evaluate on held-out sets
        for name, ds in eval_sets.items():
            tok_ds = tokenize_and_align_dataset(ds)
            pred = trainer2.predict(tok_ds)
            metrics_report[name] = compute_metrics((pred.predictions, pred.label_ids))

    else:  # joint
        # Curriculum: phase 1 exclude pseudo, phase 2 include pseudo
        def build_joint_train(include_pseudo: bool) -> Dataset:
            toks: List[List[str]] = []
            labs: List[List[int]] = []
            flags: List[bool] = []
            # PII train
            if 'train' not in pii_enc:
                raise RuntimeError('PII train split missing.')
            toks += pii_enc['train'][0]
            labs += pii_enc['train'][1]
            flags += [True] * len(pii_enc['train'][0])
            base_pii_count = len(pii_enc['train'][0])
            # Pseudo
            if include_pseudo and (not args.disable_pseudo) and pseudo_tokens:
                pt, pl = pseudo_tokens, pseudo_enc_labels
                pt, pl = cap_by_ratio(pt, pl, args.pseudo_max_ratio, base_pii_count)
                toks += pt
                labs += pl
                flags += [True] * len(pt)
            # CoNLL train
            if (not args.disable_conll) and 'train' in conll_enc:
                ct, cl = conll_enc['train']
                ct, cl = cap_by_ratio(ct, cl, args.conll_max_ratio, base_pii_count)
                toks += ct
                labs += cl
                flags += [False] * len(ct)
            # Oversample PII
            if args.pii_oversample_factor and args.pii_oversample_factor > 1.0:
                base_idx = [i for i, f in enumerate(flags) if f]
                base_toks = [toks[i] for i in base_idx]
                base_labs = [labs[i] for i in base_idx]
                base_flags = [True] * len(base_idx)
                extra_tokens = []
                extra_labels = []
                extra_flags = []
                reps = int(args.pii_oversample_factor) - 1
                frac = args.pii_oversample_factor - int(args.pii_oversample_factor)
                for _ in range(max(0, reps)):
                    extra_tokens.extend(base_toks)
                    extra_labels.extend(base_labs)
                    extra_flags.extend(base_flags)
                if frac > 0 and base_toks:
                    take = int(len(base_toks) * frac)
                    extra_tokens.extend(base_toks[:take])
                    extra_labels.extend(base_labs[:take])
                    extra_flags.extend(base_flags[:take])
                toks += extra_tokens
                labs += extra_labels
                flags += extra_flags
            # Apply augmentation before building dataset
            toks, labs, flags = _apply_augmentation(toks, labs, flags, label2id, args.augment)
            return make_hf_dataset(toks, labs, flags)

        # Phase 1
        total_epochs = args.epochs
        if args.curriculum_epochs and args.curriculum_epochs > 0:
            train_phase1 = build_joint_train(include_pseudo=False)
            eval_ds = eval_sets['pii_val'] if 'pii_val' in eval_sets else (eval_sets['conll_valid'] if 'conll_valid' in eval_sets else next(iter(eval_sets.values())))
            tok_train1 = tokenize_and_align_dataset(train_phase1)
            tok_eval = tokenize_and_align_dataset(eval_ds)
            trainer = make_trainer(model, tok_train1, tok_eval, args.curriculum_epochs)
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
            model = trainer.model
            remaining_epochs = max(0, total_epochs - args.curriculum_epochs)
        else:
            remaining_epochs = total_epochs

        # Phase 2 (main)
        train_phase2 = build_joint_train(include_pseudo=True)
        eval_ds = eval_sets['pii_val'] if 'pii_val' in eval_sets else (eval_sets['conll_valid'] if 'conll_valid' in eval_sets else next(iter(eval_sets.values())))
        tok_train2 = tokenize_and_align_dataset(train_phase2)
        tok_eval = tokenize_and_align_dataset(eval_ds)
        trainer = make_trainer(model, tok_train2, tok_eval, max(1, remaining_epochs))
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        model = trainer.model
        final_trainer = trainer

        # Evaluate on held-out sets
        for name, ds in eval_sets.items():
            tok_ds = tokenize_and_align_dataset(ds)
            pred = trainer.predict(tok_ds)
            metrics_report[name] = compute_metrics((pred.predictions, pred.label_ids))

    # Save model and tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    if final_trainer is not None:
        final_trainer.save_model(args.output_dir)
    else:
        # Fallback to direct save if trainer reference is unavailable
        model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Persist label mappings
    with open(os.path.join(args.output_dir, 'labels.txt'), 'w') as f:
        for l in label_list:
            f.write(l + '\n')
    try:
        with open(os.path.join(args.output_dir, 'label2id.json'), 'w') as f:
            json.dump(label2id, f, indent=2)
        with open(os.path.join(args.output_dir, 'id2label.json'), 'w') as f:
            json.dump({int(k) if isinstance(k, (int, str)) and str(k).isdigit() else k: v for k, v in id2label.items()}, f, indent=2)
    except Exception as e:
        print(f'Warning: failed to save label mappings: {e}')

    # Optional exports
    if args.export_int8:
        try:
            from torch.quantization import quantize_dynamic
            model.eval()
            cpu_model = model.to('cpu')
            qmodel = quantize_dynamic(cpu_model, {torch.nn.Linear}, dtype=torch.qint8)
            torch.save(qmodel.state_dict(), os.path.join(args.output_dir, 'model.int8.pt'))
            print('Saved dynamic-quantized model to model.int8.pt')
        except Exception as e:
            print(f'INT8 export failed: {e}')
    if args.export_onnx:
        try:
            model.eval()
            model_cpu = model.to('cpu')
            dummy = tokenizer([["hello"]], is_split_into_words=True, return_tensors="pt")
            # Ensure tensors are on CPU for export
            dummy = {k: v.to('cpu') for k, v in dummy.items()}
            model_inputs = (dummy['input_ids'], dummy['attention_mask'])
            input_names = ['input_ids','attention_mask']
            dynamic_axes = {'input_ids':{0:'batch',1:'seq'}, 'attention_mask':{0:'batch',1:'seq'}, 'logits':{0:'batch',1:'seq'}}
            if 'token_type_ids' in dummy:
                model_inputs = (dummy['input_ids'], dummy['attention_mask'], dummy['token_type_ids'])
                input_names.append('token_type_ids')
                dynamic_axes['token_type_ids'] = {0:'batch',1:'seq'}
            with torch.inference_mode():
                torch.onnx.export(
                    model_cpu,
                    model_inputs,
                    os.path.join(args.output_dir, 'model.onnx'),
                    input_names=input_names,
                    output_names=['logits'],
                    opset_version=args.onnx_opset,
                    dynamic_axes=dynamic_axes,
                )
            print('Saved ONNX model to model.onnx')
        except Exception as e:
            print(f'ONNX export failed: {e}')

    # Save metrics
    try:
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as mf:
            json.dump(metrics_report, mf, indent=2)
        print('Saved metrics.json with evaluation metrics by split.')
    except Exception as e:
        print(f'Could not save metrics.json: {e}')

    print(f"Model saved to {args.output_dir}. Set LOCAL_NER_MODEL={args.output_dir} to enable local layer.")


if __name__ == '__main__':
    main()
