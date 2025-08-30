import os
import re
import importlib
import json
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from collections import defaultdict
import logging
from pathlib import Path

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Holders for Models & Configs ---
_SPACY_NLP: Optional[Any] = None
_REGEX_PATTERNS_COMPILED: List[tuple] = []
_LABEL_SYNONYMS: Dict[str, List[str]] = {}
_CONTEXT_CAPTURE_SPECS: List[Dict] = []
_REVERSE_SYNONYM_MAP: Dict[str, str] = {} # For fast canonical label lookup

# --- Environment Variable Settings ---
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm").strip()

# --- Pydantic Models ---
class Entity(BaseModel):
    type: str
    placeholder: str
    value_preview: str
    confidence: Optional[float] = None

class RedactionResult(BaseModel):
    redacted_text: str
    entities: List[Entity]
    placeholder_to_original: Dict[str, str]

class _MatchSpan(BaseModel):
    start: int
    end: int
    text: str
    label: str
    source: str = ""
    score: Optional[float] = None
    specificity: int = 50

# --- Helper Functions ---
def _load_json_config(file_env_var: str, default_filename: str) -> Optional[Any]:
    file_path = os.getenv(file_env_var)
    base_path = Path(__file__).parent
    final_path = Path(file_path) if file_path and Path(file_path).exists() else base_path / default_filename
    if not final_path.exists():
        logger.warning(f"Configuration file not found: {final_path}")
        return None
    try:
        with open(final_path, 'r', encoding='utf-8') as f:
            logger.info(f"✅ Successfully loaded configuration from: {final_path}")
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load or parse config {final_path}: {e}")
        return None

# --- Model & Configuration Loading ---
def initialize_models():
    """Pre-loads all models and configurations specified in the environment."""
    global _REGEX_PATTERNS_COMPILED, _LABEL_SYNONYMS, _CONTEXT_CAPTURE_SPECS, _REVERSE_SYNONYM_MAP
    global _SPACY_NLP
    logger.info("🚀 Initializing PII detection models and configurations...")

    _LABEL_SYNONYMS = _load_json_config("LABEL_SYNONYMS_FILE", "label_synonyms.json") or {}
    for canonical, synonyms in _LABEL_SYNONYMS.items():
        for synonym in synonyms:
            _REVERSE_SYNONYM_MAP[synonym.upper()] = canonical
        _REVERSE_SYNONYM_MAP[canonical.upper()] = canonical

    canonical_regex_rules = _load_json_config("CANON_REGEX_FILE", "canonical_regex.json") or []
    _CONTEXT_CAPTURE_SPECS = _load_json_config("CONTEXT_CAPTURE_FILE", "context_capture.json") or []

    if canonical_regex_rules:
        patterns = []
        for rule in canonical_regex_rules:
            try:
                patterns.append((re.compile(rule["pattern"]), rule["label"], rule.get("specificity", 50)))
            except re.error as e:
                logger.warning(f"Invalid regex for label {rule.get('label')}: {e}")
        _REGEX_PATTERNS_COMPILED = sorted(patterns, key=lambda x: x[2], reverse=True)
        logger.info(f"✅ Compiled {len(_REGEX_PATTERNS_COMPILED)} regex patterns.")

    if SPACY_MODEL and importlib.util.find_spec("spacy"):
        try:
            spacy = importlib.import_module("spacy")
            _SPACY_NLP = spacy.load(SPACY_MODEL)
            logger.info(f"✅ spaCy model loaded: {SPACY_MODEL}")
        except Exception as e:
            logger.error(f"❌ Failed to load spaCy model '{SPACY_MODEL}'. Please download it. Error: {e}")
    logger.info("🎯 System initialization complete.")

# --- Detection & Redaction Logic ---
def _find_all_spans(text: str) -> List[_MatchSpan]:
    """
    Gathers findings from all configured sources with enhanced contextual analysis.
    """
    spans: List[_MatchSpan] = []
    
    # 1. Regex Detection (High confidence, specific patterns)
    for regex, label, specificity in _REGEX_PATTERNS_COMPILED:
        for match in regex.finditer(text):
            spans.append(_MatchSpan(start=match.start(), end=match.end(), text=match.group(0), label=label, source="regex", specificity=specificity))

    # 2. Context Capture Detection (Keyword-driven, high confidence)
    lower_text = text.lower()
    for spec in _CONTEXT_CAPTURE_SPECS:
        try:
            value_regex = re.compile(spec["value_regex"])
            for key in spec["keys"]:
                for match in re.finditer(re.escape(key), lower_text):
                    search_start = match.end()
                    search_window = text[search_start : search_start + spec["window"]]
                    value_match = value_regex.search(search_window)
                    if value_match:
                        start = search_start + value_match.start()
                        end = search_start + value_match.end()
                        spans.append(_MatchSpan(start=start, end=end, text=text[start:end], label=spec["label"], source="context", specificity=85))
        except re.error as e:
            logger.warning(f"Invalid context regex for label {spec.get('label')}: {e}")

    # 3. SpaCy NER Detection with Contextual Boosting (More intelligent)
    if _SPACY_NLP:
        # Define keywords that increase confidence in an ORG detection
        ORG_KEYWORDS = {
            "payment to", "working for", "partnered with", "employed by",
            "invoice from", "submitted to", "agent is from", "company",
            "solutions", "innovate", "realty"
        }
        
        try:
            doc = _SPACY_NLP(text)
            for ent in doc.ents:
                label = _REVERSE_SYNONYM_MAP.get(ent.label_.upper(), ent.label_.upper())
                
                # Default specificity for a generic spaCy entity
                specificity_score = 60
                
                # --- INTELLIGENCE BOOST FOR ORG ENTITIES ---
                if label == "ORG":
                    # Look for keywords in a 20-character window before the entity
                    window_start = max(0, ent.start_char - 20)
                    preceding_text = text[window_start:ent.start_char].lower()
                    
                    # If a keyword is found, boost the confidence (specificity)
                    if any(key in preceding_text for key in ORG_KEYWORDS):
                        specificity_score = 80  # Boosted score for high confidence
                    # Also check if the entity text itself contains a keyword
                    elif any(key in ent.text.lower() for key in ORG_KEYWORDS):
                        specificity_score = 80

                spans.append(_MatchSpan(
                    start=ent.start_char,
                    end=ent.end_char,
                    text=ent.text,
                    label=label,
                    source="spacy",
                    specificity=specificity_score # Use the new dynamic score
                ))
        except Exception as e:
            logger.error(f"Error during spaCy inference: {e}")
            
    return spans

def _adjudicate_spans(spans: List[_MatchSpan], text: str) -> List[_MatchSpan]:
    """
    Adjudication engine that resolves overlaps, filters false positives,
    and merges adjacent entities like PERSON and ADDRESS.
    """
    if not spans:
        return []

    spans.sort(key=lambda s: (s.start, -s.end))
    
    # Stage 1: Resolve Overlaps
    non_overlapping_spans: List[_MatchSpan] = []
    i = 0
    while i < len(spans):
        current_span = spans[i]
        overlapping_group = [current_span]
        max_end = current_span.end
        j = i + 1
        while j < len(spans) and spans[j].start < max_end:
            overlapping_group.append(spans[j])
            max_end = max(max_end, spans[j].end)
            j += 1
        best_span = max(overlapping_group, key=lambda s: (s.specificity, (s.end - s.start)))
        non_overlapping_spans.append(best_span)
        i = j

    # Stage 2: Filter Known False Positives
    known_false_positives = {"ORG": {"nric", "cvc", "ssn", "iban"}}
    filtered_spans: List[_MatchSpan] = []
    for span in non_overlapping_spans:
        label_upper = span.label.upper()
        text_lower = span.text.lower().strip()
        if label_upper in known_false_positives and text_lower in known_false_positives[label_upper]:
            continue
        filtered_spans.append(span)

    if not filtered_spans:
        return []
        
    # Stage 3: Merge Adjacent Entities (UPGRADED)
    merged_spans: List[_MatchSpan] = []
    current_merge = filtered_spans[0] 

    for i in range(1, len(filtered_spans)):
        next_span = filtered_spans[i] 
        gap_text = text[current_merge.end:next_span.start]
        
        # Condition to merge PERSON entities
        is_person_merge = current_merge.label == "PERSON" and next_span.label == "PERSON"
        is_short_gap = len(gap_text) < 7
        is_connector_word = is_short_gap and gap_text.strip().islower() and gap_text.strip().isalpha()
        
        # Condition to merge ADDRESS/LOCATION entities
        is_address_merge = current_merge.label in {"ADDRESS_LINE", "LOCATION"} and next_span.label in {"ADDRESS_LINE", "LOCATION"}
        is_address_gap = re.match(r"^[,\s]*$", gap_text) and len(gap_text) < 5

        if is_person_merge and (re.match(r"^[ .'-]*$", gap_text) or is_connector_word):
            current_merge.end = next_span.end
            current_merge.text = text[current_merge.start:current_merge.end]
            current_merge.specificity = max(current_merge.specificity, next_span.specificity)
        elif is_address_merge and is_address_gap:
            # Merge address parts, prioritizing ADDRESS_LINE as the final label
            current_merge.end = next_span.end
            current_merge.text = text[current_merge.start:current_merge.end]
            current_merge.specificity = max(current_merge.specificity, next_span.specificity)
            if next_span.label == "ADDRESS_LINE":
                current_merge.label = "ADDRESS_LINE"
        else:
            merged_spans.append(current_merge)
            current_merge = next_span
    
    merged_spans.append(current_merge)

    logger.info(f"Adjudication reduced {len(spans)} potential spans to {len(merged_spans)} final entities.")
    return merged_spans

def redact_text(text: str, **kwargs) -> RedactionResult:
    """Main function to find, adjudicate, and redact PII from text."""
    all_spans = _find_all_spans(text)
    final_spans = _adjudicate_spans(all_spans, text)

    entities, placeholder_map, counters = [], {}, defaultdict(int)
    redacted_parts, cursor = [], 0
    
    final_spans.sort(key=lambda s: s.start)

    for span in final_spans:
        label_upper = span.label.upper()
        canonical_label = _REVERSE_SYNONYM_MAP.get(label_upper, label_upper)
        
        if canonical_label == "IGNORE" or span.start < cursor:
            continue

        redacted_parts.append(text[cursor:span.start])
        counters[canonical_label] += 1
        placeholder = f"***[{canonical_label}#{counters[canonical_label]}]***"
        redacted_parts.append(placeholder)
        
        placeholder_map[placeholder] = span.text
        entities.append(Entity(
            type=canonical_label,
            placeholder=placeholder,
            value_preview=(span.text[:10] + "..." if len(span.text) > 10 else span.text),
            confidence=span.score
        ))
        cursor = span.end
    
    redacted_parts.append(text[cursor:])

    return RedactionResult(redacted_text="".join(redacted_parts), entities=entities, placeholder_to_original=placeholder_map)

# get_ner_status function can remain as-is
def get_ner_status() -> Dict[str, object]:
    """Returns the status of loaded models and configurations."""
    return {
        "spacy_model_loaded": _SPACY_NLP is not None,
        "spacy_model_name": SPACY_MODEL,
        "regex_patterns_loaded": len(_REGEX_PATTERNS_COMPILED),
        "label_synonyms_loaded": len(_LABEL_SYNONYMS)
    }

