"""Shared runtime utilities for deterministic and file-system-safe runs."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def ensure_dir(path: Path) -> Path:
    """Create a directory when it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The same path, after ensuring it exists.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments.

    Args:
        seed: Seed used by Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(requested_device: str) -> torch.device:
    """Choose the execution device.

    Args:
        requested_device: ``auto``, ``cpu``, ``cuda``, or any PyTorch device string.

    Returns:
        PyTorch device selected for the run.
    """
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a dictionary to disk as pretty JSON.

    Args:
        path: Destination JSON file.
        data: JSON-serializable dictionary.
    """
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON dictionary.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def to_project_path(path: Path) -> str:
    """Return a stable string representation for logs and CSV files.

    Args:
        path: Path to serialize.

    Returns:
        POSIX-like path string for reproducible artifacts.
    """
    return path.as_posix()
