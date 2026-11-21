"""ID-level estimates and paired uncertainty for the fixed SKAID pilot."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .common import ResearchError
from .skaid_pilot_common import DRAWS, ENCODERS, KS, SEEDS

FAMILIES = ("mfcc", "beats", "pretrained", "random")


def family(encoder):
    return encoder.split("_")[0]


def estimate(values, indices):
    values = np.asarray(values, dtype=np.float64)
    bootstrap = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "sd_ids": float(values.std(ddof=1)),
        "ci95_id_bootstrap": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
        "minimum_id_mean": float(values.min()),
        "maximum_id_mean": float(values.max()),
        "ids": len(values),
    }


def analyse(trials, zero_shot):
    actual = [(t["encoder"], t["subject"], t["k"], t["draw"], t["permuted"]) for t in trials]
    expected = {(e, s, k, d, False) for e in ENCODERS for s in range(17) for k in KS for d in DRAWS}
    expected |= {(e, s, 10, d, True) for e in ENCODERS for s in range(17) for d in DRAWS}
    if (
        len(set(actual)) != len(actual)
        or set(actual) != expected
        or any(t["query"]["examples"] != 60 for t in trials)
    ):
        raise ResearchError("Cakupan kondisi tidak lengkap/berulang.")
    zero_keys = [(r["source_seed"], r["subject"]) for r in zero_shot]
    if len(set(zero_keys)) != 51 or set(zero_keys) != {
        (seed, s) for seed in SEEDS for s in range(17)
    }:
        raise ResearchError("K0 tidak lengkap/berulang.")
    indices = np.random.default_rng(7471).integers(0, 17, size=(10000, 17))
    groups, per_id = [], []
    accuracy = {}
    for f in FAMILIES:
        for permuted, ks in ((False, KS), (True, (10,))):
            for k in ks:
                rows = [
                    r
                    for r in trials
                    if family(r["encoder"]) == f and r["k"] == k and r["permuted"] == permuted
                ]
                current = []
                for subject in range(17):
                    subset = [r for r in rows if r["subject"] == subject]
                    metrics = {
                        key: float(np.mean([r["query"][key] for r in subset]))
                        for key in ("accuracy", "macro_f1", "top3")
                    }
                    row = {
                        "family": f,
                        "k": k,
                        "permuted": permuted,
                        "subject": subject,
                        "session_token": subset[0]["session_token"],
                        "predictors_averaged": len(subset),
                        "support_fit_label_accuracy": float(
                            np.mean([r["support_fit_label_accuracy"] for r in subset])
                        ),
                        "support_true_label_accuracy": float(
                            np.mean([r["support_true_label_accuracy"] for r in subset])
                        ),
                        **metrics,
                    }
                    current.append(row)
                    per_id.append(row)
                values = np.array([r["accuracy"] for r in current])
                accuracy[f, k, permuted] = values
                groups.append(
                    {
                        "family": f,
                        "k": k,
                        "permuted": permuted,
                        "accuracy": estimate(values, indices),
                        "macro_f1": float(np.mean([r["macro_f1"] for r in current])),
                        "top3": float(np.mean([r["top3"] for r in current])),
                        "support_fit_label_accuracy": float(
                            np.mean([r["support_fit_label_accuracy"] for r in current])
                        ),
                        "support_true_label_accuracy": float(
                            np.mean([r["support_true_label_accuracy"] for r in current])
                        ),
                    }
                )
    comparisons = []
    for before, after in combinations(FAMILIES, 2):
        for k in KS:
            differences = accuracy[before, k, False] - accuracy[after, k, False]
            comparisons.append(
                {
                    "first": before,
                    "second": after,
                    "k": k,
                    "first_minus_second": estimate(differences, indices),
                }
            )
    zero_ids = [
        float(np.mean([r["query"]["accuracy"] for r in zero_shot if r["subject"] == s]))
        for s in range(17)
    ]
    thresholds = {
        f: {
            str(target): next((k for k in KS if accuracy[f, k, False].mean() >= target), None)
            for target in (0.8, 0.9)
        }
        for f in FAMILIES
    }
    return {
        "groups": groups,
        "per_id": per_id,
        "paired_comparisons": comparisons,
        "source_zero_shot": estimate(zero_ids, indices),
        "thresholds_on_fixed_grid": thresholds,
        "bootstrap": {
            "unit": "ID",
            "resamples": 10000,
            "seed": 7471,
            "conditional_on_selected_ids": True,
        },
    }
