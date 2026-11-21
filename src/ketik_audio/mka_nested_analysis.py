"""Paired outer-fold summaries for a fixed baseline and source-only alpha selection."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .common import ResearchError, read_json
from .mka import PLATFORMS
from .mka_lopo import ENCODERS, verified_records
from .mka_lopo_analysis import family_summary, moments, pool
from .mka_nested import ALPHAS, BASELINE, baseline_lookup


def analyse(root, result):
    rows = result["conditions"]
    expected = {(p, e, n) for p in PLATFORMS for e in ENCODERS for n in (False, True)}
    actual = {(r["heldout"], r["encoder"], r["permuted"]) for r in rows}
    if result["status"] != "complete" or len(rows) != 48 or actual != expected:
        raise ResearchError("Analisis memerlukan seluruh 48 head luar unik.")
    if any(r["selected_alpha"] not in ALPHAS for r in rows):
        raise ResearchError("Pilihan alpha bukan anggota grid terkunci.")
    records = verified_records()
    previous = baseline_lookup(read_json(BASELINE / "results.json"))
    base_pool = pool(BASELINE, list(previous.values()), records)
    selected_pool = pool(root, rows, records)
    comparisons, per_folder, changes = [], [], []
    for permuted in (False, True):
        for family in ("beats", "resnet"):
            base = [r for r in base_pool if r["permuted"] == permuted and r["family"] == family]
            selected = [
                r for r in selected_pool if r["permuted"] == permuted and r["family"] == family
            ]
            paired = [
                r["test"]["accuracy"]
                - next(b["test"]["accuracy"] for b in base if b["encoder"] == r["encoder"])
                for r in selected
            ]
            comparisons.append(
                {
                    "family": family,
                    "permuted": permuted,
                    "baseline": family_summary(base),
                    "nested": family_summary(selected),
                    "paired_accuracy_difference": moments(paired),
                }
            )
        for heldout in PLATFORMS:
            selected = [r for r in rows if r["heldout"] == heldout and r["permuted"] == permuted]
            entry = {"heldout": heldout, "permuted": permuted}
            for family in ("beats", "resnet"):
                family_rows = [r for r in selected if r["family"] == family]
                entry[family] = {
                    "baseline_accuracy": moments(
                        [r["baseline_test"]["accuracy"] for r in family_rows]
                    ),
                    "nested_accuracy": moments([r["test"]["accuracy"] for r in family_rows]),
                    "paired_difference": moments([r["difference_accuracy"] for r in family_rows]),
                    "selected_alphas": [r["selected_alpha"] for r in family_rows],
                    "inner_selected_accuracy": moments(
                        [r["inner_selected_accuracy"] for r in family_rows]
                    ),
                    "support_accuracy": moments([r["support_accuracy"] for r in family_rows]),
                }
            per_folder.append(entry)
    for row in rows:
        prior = previous[row["heldout"], row["encoder"], row["permuted"]]
        with np.load(root / row["artifact"], allow_pickle=False) as selected:
            with np.load(BASELINE / prior["artifact"], allow_pickle=False) as base:
                if not np.array_equal(selected["test"], base["test"]):
                    raise ResearchError("Prediksi berpasangan memakai indeks berbeda.")
                y = selected["test_y"]
                before, after = base["scores"].argmax(1), selected["scores"].argmax(1)
                fixed = int(((before != y) & (after == y)).sum())
                broken = int(((before == y) & (after != y)).sum())
                if fixed - broken != row["test"]["correct"] - row["baseline_test"]["correct"]:
                    raise ResearchError("Perubahan prediksi tidak cocok dengan selisih akurasi.")
                changes.append(
                    {
                        "id": row["id"],
                        "heldout": row["heldout"],
                        "encoder": row["encoder"],
                        "permuted": row["permuted"],
                        "changed_predictions": int((before != after).sum()),
                        "new_correct": fixed,
                        "new_errors": broken,
                        "net_correct": fixed - broken,
                    }
                )
    edges = []
    for permuted in (False, True):
        subset = [r for r in rows if r["permuted"] == permuted]
        edges.append(
            {
                "permuted": permuted,
                "count": len(subset),
                "alpha_histogram": dict(Counter(str(r["selected_alpha"]) for r in subset)),
                "lower_grid_edge": sum(r["selected_alpha"] == min(ALPHAS) for r in subset),
                "upper_grid_edge": sum(r["selected_alpha"] == max(ALPHAS) for r in subset),
            }
        )
    return {
        "comparisons": comparisons,
        "per_folder": per_folder,
        "baseline_pooled": base_pool,
        "nested_pooled": selected_pool,
        "paired_prediction_changes": changes,
        "alpha_choices": edges,
        "unique_test_clips_per_configuration": 1296,
        "unique_test_parent_wavs_per_configuration": 216,
        "variation_scope": "one BEATs; sample SD across 3 ResNet initializations, not sessions",
        "outer_scores_for_unselected_alphas_computed": False,
    }
