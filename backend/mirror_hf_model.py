#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Optional


def parse_args():
	p = argparse.ArgumentParser(description="Mirror a Hugging Face model/tokenizer locally for offline use.")
	p.add_argument("model_id", help="HF model id or local path (e.g., dslim/bert-base-NER)")
	p.add_argument("output_dir", help="Directory to save the mirrored model")
	p.add_argument("--revision", default=None, help="Optional model revision/tag/commit")
	p.add_argument("--trust-remote-code", action="store_true", help="Allow loading models with custom code")
	return p.parse_args()


def ensure_transformers() -> None:
	try:
		import transformers  # noqa: F401
	except Exception:
		print("Transformers is required. Please 'pip install transformers'.", file=sys.stderr)
		sys.exit(1)


def mirror(model_id: str, out_dir: str, revision: Optional[str], trust_remote_code: bool) -> None:
	from transformers import AutoModelForTokenClassification, AutoTokenizer
	os.makedirs(out_dir, exist_ok=True)
	print(f"Mirroring model={model_id} -> {out_dir}")
	kwargs = {}
	if revision:
		kwargs["revision"] = revision
	if trust_remote_code:
		kwargs["trust_remote_code"] = True
	# Tokenizer
	tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, **kwargs)
	tok.save_pretrained(out_dir)
	# Model
	mdl = AutoModelForTokenClassification.from_pretrained(model_id, **kwargs)
	mdl.save_pretrained(out_dir)
	print("Done.")


def main():
	args = parse_args()
	ensure_transformers()
	mirror(args.model_id, args.output_dir, args.revision, args.trust_remote_code)


if __name__ == "__main__":
	main()
