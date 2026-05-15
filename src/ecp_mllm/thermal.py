from __future__ import annotations

from typing import Iterable


THERMAL_COARSE_LABELS: tuple[str, ...] = (
    "false_positive",
    "bird",
    "rodent",
    "possum",
    "cat",
    "hedgehog",
    "mustelid",
    "other",
)

_FALSE_POSITIVE_TERMS = (
    "false positive",
    "false_positive",
    "false-positive",
    "falsepositive",
    "fp",
    "nothing",
    "empty",
    "unknown",
    "unidentified",
    "insect",
)
_BIRD_TERMS = ("bird", "kiwi", "kea", "weka", "morepork", "owl", "pukeko", "gull")
_RODENT_TERMS = ("rodent", "rat", "mouse", "mice")
_POSSUM_TERMS = ("possum",)
_CAT_TERMS = ("cat", "feline", "feral cat")
_HEDGEHOG_TERMS = ("hedgehog",)
_MUSTELID_TERMS = ("mustelid", "stoat", "ferret", "weasel")


def _normalize_label(label: str | None) -> str:
    return str(label or "").strip().lower().replace("-", " ").replace("_", " ")


def map_thermal_label_to_coarse(label: str | None) -> str:
    normalized = _normalize_label(label)
    if not normalized:
        return "other"
    if any(term in normalized for term in _FALSE_POSITIVE_TERMS):
        return "false_positive"
    if any(term in normalized for term in _BIRD_TERMS):
        return "bird"
    if any(term in normalized for term in _RODENT_TERMS):
        return "rodent"
    if any(term in normalized for term in _POSSUM_TERMS):
        return "possum"
    if any(term in normalized for term in _CAT_TERMS):
        return "cat"
    if any(term in normalized for term in _HEDGEHOG_TERMS):
        return "hedgehog"
    if any(term in normalized for term in _MUSTELID_TERMS):
        return "mustelid"
    if normalized in THERMAL_COARSE_LABELS:
        return normalized
    return "other"


def collapse_thermal_labels(labels: Iterable[str] | None) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_label in labels or ():
        coarse = map_thermal_label_to_coarse(raw_label)
        if coarse not in seen:
            seen.add(coarse)
            ordered.append(coarse)
    return ordered


def infer_false_positive_from_labels(labels: Iterable[str] | None) -> bool:
    coarse = collapse_thermal_labels(labels)
    return bool(coarse) and all(label == "false_positive" for label in coarse)
