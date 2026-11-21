"""Descriptive paired analysis; rejects incomplete or duplicate pilot conditions."""

from __future__ import annotations

import numpy as np

from .common import ResearchError
from .resnet_fewshot import BLOCKS, KS


def moments(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1))}


def analyse(record):
    expected = {
        (block, method, k, seed, draw)
        for block, (seed, draw) in enumerate(BLOCKS, start=1)
        for method in ("scratch", "adapt")
        for k in KS
    }
    expected |= {
        (block, "source_only", 0, seed, None) for block, (seed, _) in enumerate(BLOCKS, start=1)
    }
    actual = {
        (row["block"], row["method"], row["k"], row["training_seed"], row["support_seed"])
        for row in record["conditions"]
    }
    if (
        record["status"] != "complete"
        or len(record["conditions"]) != len(expected)
        or actual != expected
    ):
        raise ResearchError("Analisis memerlukan seluruh 33 kondisi unik yang ditetapkan.")
    lookup = {(r["block"], r["method"], r["k"]): r for r in record["conditions"]}
    groups, paired = [], []
    for k in KS:
        for method in ("scratch", "adapt"):
            rows = [lookup[block, method, k] for block in range(1, 4)]
            groups.append(
                {
                    "method": method,
                    "k": k,
                    "target_fit_labels": 36 * k,
                    "accuracy": moments([r["test"]["accuracy"] for r in rows]),
                    "macro_f1": moments([r["test"]["macro_f1"] for r in rows]),
                    "top5": moments([r["test"]["top5"] for r in rows]),
                    "support_accuracy": moments([r["support_accuracy"] for r in rows]),
                    "minimum_support_accuracy": min(r["support_accuracy"] for r in rows),
                    "training_seconds": moments([r["training_seconds"] for r in rows]),
                    "per_block_accuracy": [r["test"]["accuracy"] for r in rows],
                }
            )
        differences = [
            lookup[block, "adapt", k]["test"]["accuracy"]
            - lookup[block, "scratch", k]["test"]["accuracy"]
            for block in range(1, 4)
        ]
        paired.append(
            {
                "k": k,
                "difference": moments(differences),
                "per_block_difference": differences,
                "positive_blocks": sum(d > 0 for d in differences),
                "negative_blocks": sum(d < 0 for d in differences),
                "tied_blocks": sum(d == 0 for d in differences),
            }
        )
    thresholds = {}
    for method in ("scratch", "adapt"):
        rows = [r for r in groups if r["method"] == method]
        thresholds[method] = {
            str(threshold): {
                "first_mean_k": next(
                    (r["k"] for r in rows if r["accuracy"]["mean"] >= threshold / 100), None
                ),
                "first_all_blocks_k": next(
                    (r["k"] for r in rows if min(r["per_block_accuracy"]) >= threshold / 100), None
                ),
            }
            for threshold in (80, 90)
        }
    zero = [r for r in record["conditions"] if r["method"] == "source_only"]
    return {
        "source": "zoom",
        "target": "phone",
        "groups": groups,
        "paired": paired,
        "observed_grid_thresholds": thresholds,
        "zero_shot_accuracy": moments([r["test"]["accuracy"] for r in zero]),
        "source_train_labels_unique": 648,
        "source_validation_labels_unique": 72,
        "source_training_seconds_total": sum(r["training_seconds"] for r in record["sources"]),
        "target_training_seconds_total": sum(
            r["training_seconds"] for r in record["conditions"] if r["k"] > 0
        ),
        "variation_scope": "three joint draws of initialization and support selection",
        "not_independent_test_event_count": 360,
    }
