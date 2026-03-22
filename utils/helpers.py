"""Shared utility functions."""

import os
import yaml
from pathlib import Path


def _load_dotenv(env_path: str = ".env") -> None:
    """Load .env file into os.environ (no third-party dependency)."""
    p = Path(env_path)
    if not p.exists():
        return
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


# Mapping: env var name -> (yaml section, yaml key)
_ENV_TO_CONFIG = {
    "AZURE_ENDPOINT": ("azure", "endpoint"),
    "AZURE_API_KEY": ("azure", "api_key"),
    "AZURE_MODEL_ID": ("azure", "model_id"),
    "GOOGLE_APPLICATION_CREDENTIALS": ("google", "credentials_path"),
    "AWS_REGION": ("aws", "region"),
    "AWS_ACCESS_KEY_ID": ("aws", "access_key_id"),
    "AWS_SECRET_ACCESS_KEY": ("aws", "secret_access_key"),
    "MISTRAL_API_KEY": ("mistral", "api_key"),
    "SARVAM_API_KEY": ("sarvam", "api_key"),
    "SARVAM_ENDPOINT": ("sarvam", "endpoint"),
}


def _overlay_env_vars(config: dict) -> dict:
    """Override YAML config values with environment variables when set."""
    for env_var, (section, key) in _ENV_TO_CONFIG.items():
        value = os.environ.get(env_var)
        if value:
            config.setdefault(section, {})[key] = value
    return config


def load_config(path: str = "configs/config.local.yaml") -> dict:
    """Load YAML config, then overlay with env vars from .env file.

    Priority (highest wins): env var > .env file > YAML config.
    """
    _load_dotenv()

    p = Path(path)
    if not p.exists():
        p = Path("configs/config.yaml")
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(p) as f:
        config = yaml.safe_load(f)

    return _overlay_env_vars(config)


def get_device(config: dict) -> str:
    """Determine compute device from config."""
    import torch
    device = config.get("device", "auto")
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def find_documents(dataset_dir: str, extensions=None) -> list:
    """Recursively find all document files in the dataset directory."""
    extensions = extensions or [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".pdf", ".bmp"]
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    files = []
    for ext in extensions:
        files.extend(dataset_path.rglob(f"*{ext}"))

    # Exclude ground_truth folder and hidden files
    files = [f for f in files
             if "ground_truth" not in str(f)
             and not f.name.startswith(".")
             and "_tmp_" not in str(f)]

    return sorted(files, key=lambda f: str(f))


def get_document_category(doc_path: str) -> str:
    """Extract category from document path (e.g., '01_printed_english/invoices')."""
    parts = Path(doc_path).parts
    for i, p in enumerate(parts):
        if p.startswith(("01_", "02_", "03_", "04_", "05_", "06_", "07_")):
            return "/".join(parts[i:i+2]) if i+1 < len(parts) else parts[i]
    return "unknown"


def get_ground_truth(doc_path: str, gt_dir: str) -> str:
    """Find ground truth text for a document, if available."""
    doc = Path(doc_path)
    gt_base = Path(gt_dir)

    # Try exact name match with _gt.txt suffix
    gt_file = gt_base / (doc.stem + "_gt.txt")
    if gt_file.exists():
        return gt_file.read_text(encoding="utf-8")

    # Try matching by searching recursively
    for gt in gt_base.rglob(f"{doc.stem}_gt.txt"):
        return gt.read_text(encoding="utf-8")
    for gt in gt_base.rglob(f"{doc.stem}.txt"):
        return gt.read_text(encoding="utf-8")
    for gt in gt_base.rglob(f"{doc.stem}.md"):
        return gt.read_text(encoding="utf-8")

    # Try JSON ground truth (from FUNSD etc.)
    for gt in gt_base.rglob(f"{doc.stem}.json"):
        import json
        data = json.loads(gt.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "form" in data:
            words = [w["text"] for w in data["form"] if "text" in w]
            return " ".join(words)
        return json.dumps(data)

    return ""
