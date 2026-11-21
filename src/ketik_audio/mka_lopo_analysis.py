"""Descriptive, factorial summaries without treating repeated BEATs as new models."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .common import ResearchError
from .mka import PLATFORMS
from .mka_lopo import DRAWS, ENCODERS, KS, parent_scores, verified_records
from .mka_zeroshot import metrics


def moments(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
        "values": values.tolist(),
        "count": len(values),
    }


def check_conditions(result):
    expected = {
        (p, k, draw, encoder, negative)
        for p in PLATFORMS
        for k in KS
        for draw in ((None,) if k == 6 else DRAWS)
        for encoder in ENCODERS
        for negative in ((False, True) if k == 6 else (False,))
    }
    rows = result["conditions"]
    actual = {(r["heldout"], r["k"], r["draw"], r["encoder"], r["permuted"]) for r in rows}
    if result["status"] != "complete" or len(rows) != 336 or actual != expected:
        raise ResearchError("Analisis memerlukan seluruh 336 kondisi unik.")


def pool(root, rows, records):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["k"], row["draw"], row["encoder"], row["permuted"]].append(row)
    y = np.asarray([r["label"] for r in records])
    pooled = []
    for (k, draw, encoder, negative), conditions in grouped.items():
        scores = np.empty((len(records), 36), dtype=np.float64)
        seen = np.zeros(len(records), dtype=np.int64)
        for row in conditions:
            with np.load(root / row["artifact"], allow_pickle=False) as saved:
                indices = saved["test"]
                if not np.array_equal(saved["test_y"], y[indices]):
                    raise ResearchError("Label prediksi berbeda dari provenance.")
                scores[indices] = saved["scores"]
                seen[indices] += 1
        if len(conditions) != 6 or not np.all(seen == 1):
            raise ResearchError("Prediksi out-of-fold bukan partisi seluruh potongan.")
        parent_y, grouped_scores, _ = parent_scores(records, np.arange(len(records)), scores)
        pooled.append(
            {
                "k": k,
                "draw": draw,
                "encoder": encoder,
                "family": "beats" if encoder == "beats" else "resnet",
                "permuted": negative,
                "source_seed": None if encoder == "beats" else int(encoder[-2:]),
                "test": metrics(y, scores),
                "parent": metrics(parent_y, grouped_scores),
                "support_accuracy_mean_over_folds": float(
                    np.mean([r["support_accuracy"] for r in conditions])
                ),
            }
        )
    return pooled


def family_summary(rows):
    summary = {}
    for unit in ("test", "parent"):
        summary[unit] = {
            name: moments([row[unit][name] for row in rows])
            for name in ("accuracy", "top5", "macro_f1")
        }
    summary["configurations"] = [
        {
            "encoder": r["encoder"],
            "draw": r["draw"],
            "accuracy": r["test"]["accuracy"],
            "correct": r["test"]["correct"],
        }
        for r in rows
    ]
    return summary


def analyse(root, result):
    check_conditions(result)
    records = verified_records()
    pooled = pool(root, result["conditions"], records)
    per_folder = []
    for heldout in PLATFORMS:
        rows = [
            r
            for r in result["conditions"]
            if r["heldout"] == heldout and r["k"] == 6 and not r["permuted"]
        ]
        beats = next(r for r in rows if r["encoder"] == "beats")
        resnet = [r for r in rows if r["family"] == "resnet"]
        per_folder.append(
            {
                "heldout": heldout,
                "beats": family_summary([beats]),
                "resnet": family_summary(resnet),
                "beats_minus_resnet_mean_accuracy": beats["test"]["accuracy"]
                - float(np.mean([r["test"]["accuracy"] for r in resnet])),
            }
        )
    curve = []
    for k in KS:
        rows = [r for r in pooled if r["k"] == k and not r["permuted"]]
        beats = [r for r in rows if r["family"] == "beats"]
        resnet = [r for r in rows if r["family"] == "resnet"]
        source_means = [
            float(np.mean([r["test"]["accuracy"] for r in resnet if r["source_seed"] == s]))
            for s in (42, 43, 44)
        ]
        draw_means, paired = [], []
        for draw in (None,) if k == 6 else DRAWS:
            baseline = float(np.mean([r["test"]["accuracy"] for r in resnet if r["draw"] == draw]))
            value = next(r["test"]["accuracy"] for r in beats if r["draw"] == draw)
            draw_means.append(baseline)
            paired.append(value - baseline)
        curve.append(
            {
                "k_per_key_per_training_folder": k,
                "fit_labels_per_fold": 180 * k,
                "fit_parent_wavs_per_fold": 180,
                "fit_labels_per_key": 5 * k,
                "beats": family_summary(beats),
                "resnet": family_summary(resnet),
                "resnet_source_mean_accuracy": moments(source_means),
                "resnet_draw_mean_accuracy": moments(draw_means),
                "paired_draw_difference": moments(paired),
            }
        )
    negative = [r for r in pooled if r["permuted"]]
    return {
        "per_folder_full_budget": per_folder,
        "curve": curve,
        "negative_control": {
            "beats": family_summary([r for r in negative if r["family"] == "beats"]),
            "resnet": family_summary([r for r in negative if r["family"] == "resnet"]),
        },
        "pooled_configurations": pooled,
        "unique_test_clips_per_pooled_configuration": 1296,
        "unique_test_parent_wavs_per_pooled_configuration": 216,
        "models_are_not_ensembled": True,
        "variation_scope": {
            "beats_curve": "three support draws; K6 one unique support set and one fixed encoder",
            "resnet_curve": "3 source initializations x 3 support draws; K6 3 initializations",
            "folds": "six observed folders sharing training data, not independent sessions",
        },
    }
