"""Locate every CPU/GPU-feature prediction disagreement without changing chosen recipes."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ketik_audio.common import file_hash, fresh_directory, now, read_json, write_json
from ketik_audio.skaid_pilot_common import CACHE, DATA, ENCODERS, KEYS, ROOT, verify
from ketik_audio.skaid_selection import OUTPUT as RUN
from ketik_audio.skaid_selection_core import RECIPES, transform

AUDIT = ROOT / "reports/skaid-selection-audit"
OUTPUT = ROOT / "reports/skaid-selection-numeric"


def detail(encoder, i, j, gpu, cpu, primary, secondary, records, labels, mode):
    recipe = RECIPES[int(primary["recipe_index"][i])]
    support = primary["support"][i]
    support = support[support >= 0]
    query = primary["query"][i]
    gp, cp = gpu[support].astype(float), cpu[support].astype(float)
    std, cpu_std = gp.std(0), cp.std(0)
    ulp = np.abs(
        np.spacing(np.maximum(np.abs(gp.mean(0)), np.finfo(np.float32).tiny).astype(np.float32))
    ).astype(float)
    close_to_quantization = (std > 1e-12) & (std <= 4 * ulp)
    xs, qs, _, _ = transform(gp, gpu[query].astype(float), recipe["normalizer"])
    cs, cq, _, _ = transform(cp, cpu[query].astype(float), recipe["normalizer"])
    row = primary["query_scores"][i, j]
    other = secondary[i, j]
    sorted_scores = np.sort(row)
    flip = int(row.argmax()) != int(other.argmax())
    return {
        "kind": mode,
        "encoder": encoder,
        "array_index": i,
        "query_position": j,
        "clip_id": records[int(query[j])]["id"],
        "true_key": KEYS[int(labels[query[j]])],
        "primary_prediction": KEYS[int(row.argmax())],
        "cpu_prediction": KEYS[int(other.argmax())],
        "prediction_changed": flip,
        "normalizer": recipe["normalizer"],
        "alpha": recipe["alpha"],
        "primary_top1_minus_top2_score": float(sorted_scores[-1] - sorted_scores[-2]),
        "max_abs_score_difference": float(np.max(np.abs(row - other))),
        "raw_support_max_abs_feature_difference": float(np.max(np.abs(gp - cp))),
        "near_constant_threshold_mask_changes": int(np.sum((std <= 1e-12) != (cpu_std <= 1e-12))),
        "active_support_features_with_sd_at_most_4_float32_ulps": int(close_to_quantization.sum()),
        "transformed_support_max_abs_difference": float(np.abs(xs - cs).max()),
        "transformed_query_max_abs_difference": float(np.abs(qs[j] - cq[j]).max()),
        "primary_centered_gram_condition_number": float(
            np.linalg.cond(
                (xs - xs.mean(0)) @ (xs - xs.mean(0)).T + recipe["alpha"] * np.eye(len(xs))
            )
        ),
    }


def run():
    audit = read_json(AUDIT / "audit.json")
    verify(AUDIT, audit)
    verify(RUN, read_json(RUN / "results.json"))
    fresh_directory(OUTPUT)
    paths = [
        Path(__file__),
        AUDIT / "audit.json",
        RUN / "results.json",
        CACHE / "manifest.json",
        ROOT / "reports/skaid-pilot-audit/audit.json",
    ]
    signature = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    write_json(OUTPUT / "lock.json", {"created_at": now(), "input_sha256": signature})
    with np.load(DATA / "segments.npz", allow_pickle=False) as z:
        labels = z["y"]
    records = read_json(DATA / "records.json")["records"]
    flips, maxima = [], []
    with (
        np.load(CACHE / "features.npz", allow_pickle=False) as main,
        np.load(ROOT / "reports/skaid-pilot-audit/cpu-features.npz", allow_pickle=False) as cpu,
    ):
        for encoder in ENCODERS:
            with np.load(RUN / "final" / f"{encoder}.npz", allow_pickle=False) as z:
                primary = {k: z[k] for k in ("query_scores", "query", "support", "recipe_index")}
            with np.load(AUDIT / f"cpu-scores-{encoder}.npz", allow_pickle=False) as z:
                secondary = z["query_scores"]
            changed = primary["query_scores"].argmax(2) != secondary.argmax(2)
            raw, control = main[encoder], cpu[encoder]
            for i, j in np.argwhere(changed):
                flips.append(
                    detail(
                        encoder,
                        int(i),
                        int(j),
                        raw,
                        control,
                        primary,
                        secondary,
                        records,
                        labels,
                        "prediction_change",
                    )
                )
            per_query_error = np.max(np.abs(primary["query_scores"] - secondary), axis=2)
            i, j = np.unravel_index(per_query_error.argmax(), per_query_error.shape)
            maxima.append(
                detail(
                    encoder,
                    int(i),
                    int(j),
                    raw,
                    control,
                    primary,
                    secondary,
                    records,
                    labels,
                    "maximum_score_error",
                )
            )
    with (OUTPUT / "prediction-changes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flips[0]))
        writer.writeheader()
        writer.writerows(flips)
    assert len(flips) == audit["cpu_feature_prediction_disagreements_repeated_trials"]
    result = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": signature,
        "prediction_changes": flips,
        "maximum_score_errors_per_encoder": maxima,
        "unique_clips_changed": len({r["clip_id"] for r in flips}),
        "unchanged_frozen_choices": True,
        "parameter_updates": 0,
    }
    result["artifact_sha256"] = {
        p.name: file_hash(p) for p in sorted(OUTPUT.iterdir()) if p.is_file()
    }
    write_json(OUTPUT / "summary.json", result)
    return result


if __name__ == "__main__":
    run()
