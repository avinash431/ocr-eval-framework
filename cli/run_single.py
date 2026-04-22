#!/usr/bin/env python3
"""Run one model on one document. Good for quick testing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from utils.helpers import load_config
from utils.runner import EvalRunner
from models import list_models


def main():
    parser = argparse.ArgumentParser(description="Run one OCR model on one document")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--input", required=True, help="Path to document image/PDF")
    parser.add_argument("--config", default="configs/config.local.yaml")
    parser.add_argument("--output", help="Save raw text to this file")
    args = parser.parse_args()

    config = load_config(args.config)
    runner = EvalRunner(config)

    print(f"\nAvailable models: {', '.join(list_models())}\n")

    result = runner.run_single(args.model, args.input)

    if args.output and result.success:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result.raw_text)
        print(f"\n💾 Output saved to: {args.output}")

    print("\n--- Full Result ---")
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
