#!/usr/bin/env python3
"""Run all (or selected) models on the entire test dataset."""

import argparse
from utils.helpers import load_config
from utils.runner import EvalRunner
from models import list_models


def main():
    parser = argparse.ArgumentParser(description="Run OCR models on all test documents")
    parser.add_argument("--models", nargs="*", help="Specific models to run (default: all)")
    parser.add_argument("--config", default="configs/config.local.yaml")
    parser.add_argument("--list", action="store_true", help="List available models and exit")
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for m in list_models():
            print(f"  - {m}")
        return

    config = load_config(args.config)
    runner = EvalRunner(config)

    available = list_models()
    models_to_run = args.models if args.models else available

    # Validate requested models
    invalid = [m for m in models_to_run if m not in available]
    if invalid:
        print(f"⚠ Unknown models: {invalid}")
        print(f"Available: {available}")
        return

    print(f"Models to run: {models_to_run}")
    run_dir = runner.run_batch(models_to_run)
    print(f"\n📁 All results saved to: {run_dir}")


if __name__ == "__main__":
    main()
