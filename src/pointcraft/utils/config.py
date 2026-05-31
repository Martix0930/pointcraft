"""Tiny config loader. YAML is the project config format (see
docs/03_EXPERIMENT_PROTOCOL.md). Kept dependency-light: just wraps yaml.safe_load
and resolves paths relative to a base directory.
"""
from __future__ import annotations

import os
from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML config file into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path!r} must be a YAML mapping, got {type(cfg).__name__}")
    return cfg


def resolve_path(path: str, base_dir: str) -> str:
    """Resolve a (possibly relative) config path against base_dir."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base_dir, path))
