#!/usr/bin/env python3
"""Run one model on the entire test dataset."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from utils.helpers import load_config
from utils.runner import EvalRunner
from models import list_models


def main():
    parser = argparse.ArgumentParser(description="Run one OCR model on all test documents")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--config", default="configs/config.local.yaml")
    parser.add_argument("--resume", help="Resume an existing run directory "
                        "(skip documents with raw_output already on disk)")
    args = parser.parse_args()

    config = load_config(args.config)

    print(f"Available models: {', '.join(list_models())}\n")

    runner = EvalRunner(config)
    run_dir = runner.run_model(args.model, resume_dir=args.resume)
    print(f"\n📁 Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
