"""Separate classification, detection, and full-sequence measurements."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .common import LABELS


def classify(truth, predictions) -> dict:
    truth, predictions = np.asarray(truth), np.asarray(predictions)
    labels = np.arange(len(LABELS))
    return {
        "count": len(truth),
        "accuracy": float(accuracy_score(truth, predictions)) if len(truth) else 0.0,
        "macro_f1": float(
            f1_score(truth, predictions, labels=labels, average="macro", zero_division=0)
        )
        if len(truth)
        else 0.0,
        "confusion_matrix": confusion_matrix(truth, predictions, labels=labels).tolist(),
        "per_class": classification_report(
            truth,
            predictions,
            labels=labels,
            target_names=list(LABELS),
            output_dict=True,
            zero_division=0,
        )
        if len(truth)
        else {},
    }


def match_events(truth, detected, tolerance: float) -> list[tuple[int, int]]:
    """One-to-one chronological matching, including missed and extra events."""
    i, j = 0, 0
    matches = []
    while i < len(truth) and j < len(detected):
        difference = detected[j] - truth[i]
        if abs(difference) <= tolerance:
            matches.append((i, j))
            i += 1
            j += 1
        elif difference < -tolerance:
            j += 1
        else:
            i += 1
    return matches


def detection(truth, detected, tolerance: float) -> dict:
    matches = match_events(truth, detected, tolerance)
    count = len(matches)
    precision = count / len(detected) if len(detected) else 0.0
    recall = count / len(truth) if len(truth) else 0.0
    return {
        "true_events": len(truth),
        "detected_events": len(detected),
        "matched": count,
        "missed": len(truth) - count,
        "extra": len(detected) - count,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tolerance_ms": tolerance * 1000,
    }


def edit_distance(truth: str, prediction: str) -> int:
    previous = list(range(len(prediction) + 1))
    for i, actual in enumerate(truth, 1):
        current = [i]
        for j, guessed in enumerate(prediction, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (actual != guessed))
            )
        previous = current
    return previous[-1]
