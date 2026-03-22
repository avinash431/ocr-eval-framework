"""Model registry — discovers and loads all model wrappers."""

import importlib
import pkgutil
from pathlib import Path
from models.base import BaseOCRModel, OCRResult

MODEL_REGISTRY = {}


def register_model(cls):
    """Decorator to register a model class."""
    instance = cls.__new__(cls)
    instance.config = {}
    MODEL_REGISTRY[instance.name] = cls
    return cls


def get_model(name: str, config: dict) -> BaseOCRModel:
    _auto_discover()
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name](config)


def list_models() -> list:
    _auto_discover()
    return sorted(MODEL_REGISTRY.keys())


def _auto_discover():
    """Import all modules in the models package to trigger @register_model."""
    if len(MODEL_REGISTRY) > 0:
        return
    pkg_dir = Path(__file__).parent
    for _, module_name, _ in pkgutil.iter_modules([str(pkg_dir)]):
        if module_name not in ("__init__", "base"):
            importlib.import_module(f"models.{module_name}")
