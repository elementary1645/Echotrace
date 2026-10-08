"""AudioSet label names -> EchoTrace categories, resolved to model output indices by NAME."""
import csv
import difflib
from pathlib import Path

import numpy as np

from echotrace import config


def load_label_names(csv_path: Path = config.LABELS_CSV) -> list[str]:
    """Display names in model output order (row i of class_labels_indices.csv = output i)."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    names = [r["display_name"] for r in sorted(rows, key=lambda r: int(r["index"]))]
    if len(names) != config.NUM_CLASSES:
        raise ValueError(f"{csv_path}: {len(names)} classes, expected {config.NUM_CLASSES}")
    return names


def resolve(categories: dict[str, list[str]], label_names: list[str]) -> dict[str, list[int]]:
    """Map each category to its label indices. Fails loudly listing every unknown name."""
    index = {n: i for i, n in enumerate(label_names)}
    missing = [(cat, n) for cat, members in categories.items() for n in members if n not in index]
    if missing:
        lines = []
        for cat, n in missing:
            close = difflib.get_close_matches(n, label_names, n=3)
            lines.append(f"  {cat}: '{n}' not an AudioSet label" + (f" (did you mean {close}?)" if close else ""))
        raise ValueError("Unknown AudioSet label names in config.CATEGORIES:\n" + "\n".join(lines))
    return {cat: [index[n] for n in members] for cat, members in categories.items()}


class CategoryMapper:
    """probs (527,) -> {category: (score, best_label)} with score = max over member labels."""

    def __init__(self, categories: dict[str, list[str]] = config.CATEGORIES,
                 label_names: list[str] | None = None):
        self.label_names = label_names or load_label_names()
        self.indices = resolve(categories, self.label_names)

    def scores(self, probs: np.ndarray) -> dict[str, tuple[float, str]]:
        out = {}
        for cat, idx in self.indices.items():
            j = idx[int(np.argmax(probs[idx]))]
            out[cat] = (float(probs[j]), self.label_names[j])
        return out

    def top_k(self, probs: np.ndarray, k: int = config.TOP_K) -> list[tuple[str, float]]:
        order = np.argsort(probs)[::-1][:k]
        return [(self.label_names[i], float(probs[i])) for i in order]
