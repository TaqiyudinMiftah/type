"""Development comparison of timestamp-conditioned crops with fixed event splits.

Run as python -m research.run_loq_crop_robustness --output <new-directory>.
Select by mean macro accuracy on source validation over three crop shifts.
The historical recording remains exploratory and never selects a model.
"""

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone

from ketik_audio.common import file_hash, fresh_directory, read_json, write_json
from ketik_audio.skaid_pilot_features import mfcc_features
from research.run_loq_transfer import (
    CLASSES,
    HISTORICAL,
    RATE,
    RUN,
    SOURCE,
    measure,
    metadata,
    recording,
    targets,
    waves,
)

TEST_OFFSETS = (-12.5, 0.0, 12.5)
FIT_OFFSETS = (-12.5, -6.25, 0.0, 6.25, 12.5)


def peak_features(record, rows, offset_ms):
    size = RATE // 200
    floor = len(record.audio) // size * size
    energy = record.audio[:floor].reshape(-1, size).std(axis=1, dtype=np.float64)
    centers = (np.arange(len(energy)) + 0.5) * size
    segments = []
    for row in rows:
        prior = row["frame"] + RATE * offset_ms / 1000
        lo, hi = np.searchsorted(centers, [prior - 0.025 * RATE, prior + 0.035 * RATE])
        peak = centers[lo + int(energy[lo:hi].argmax())]
        start = round(peak - 0.015 * RATE)
        segments.append({"start": start, "end": start + 3840})
    return mfcc_features(waves(record, segments))


def fixed_features(record, rows, offset_ms):
    return mfcc_features(waves(record, rows, round(RATE * offset_ms / 1000)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    fresh_directory(output)
    source = recording(SOURCE, metadata(SOURCE), output, "source")
    target = recording(HISTORICAL, metadata(HISTORICAL), output, "historical", exploratory=True)
    prepared = RUN / "prepared"
    checkpoint = read_json(RUN / "models/results.json")
    ridge_path = RUN / "models/mfcc-ridge.joblib"
    assert file_hash(ridge_path) == checkpoint["artifacts"][ridge_path.name]
    assert file_hash(prepared / "features.npz") == checkpoint["features_sha256"]
    ridge = joblib.load(ridge_path)
    source_rows = {row["event_seq"]: row for row in source.rows}
    folds = {"train": [], "validation": [], "test": []}
    with (prepared / "split.csv").open(newline="", encoding="utf-8") as file:
        for saved in csv.DictReader(file):
            row = source_rows[int(saved["event_seq"])]
            assert saved["key"] == row["key"]
            assert int(saved["start_frame"]) == row["start"]
            assert int(saved["end_frame"]) == row["end"]
            # All shifts and peak searches fit inside the original block guard.
            block_start = int(saved["block"]) * 8 * RATE
            assert row["start"] - 0.0375 * RATE >= block_start
            assert row["end"] + 0.0475 * RATE <= block_start + 8 * RATE
            folds[saved["split"]].append(row)
    train = folds["train"]
    y = targets(train)
    write_json(
        output / "plan.json",
        {
            "script_sha256": file_hash(Path(__file__)),
            "ridge_sha256": file_hash(ridge_path),
            "split_sha256": file_hash(prepared / "split.csv"),
            "fit_positions_ms": FIT_OFFSETS,
            "test_positions_ms": TEST_OFFSETS,
            "peak_search_ms": [-25, 35],
            "fit_unique_labels": len(train),
            "validation_labels": len(folds["validation"]),
            "source_test_labels": len(folds["test"]),
            "historical_labels": len(target.rows),
            "selection": "mean validation macro accuracy across three positions",
        },
    )
    jittered = np.concatenate([fixed_features(source, train, offset) for offset in FIT_OFFSETS])
    weights = np.full(len(jittered), 1 / len(FIT_OFFSETS))
    varied = clone(ridge).fit(
        jittered, np.tile(y, len(FIT_OFFSETS)), ridgeclassifier__sample_weight=weights
    )
    peaked = clone(ridge).fit(peak_features(source, train, 0), y)
    models = {
        "fixed": (ridge, fixed_features),
        "varied": (varied, fixed_features),
        "peak_centered": (peaked, peak_features),
    }
    results = {}
    for name, (model, features) in models.items():
        results[name] = {}
        if name != "fixed":
            joblib.dump(model, output / f"ridge-{name}.joblib")
        for partition, record, rows in (
            ("validation", source, folds["validation"]),
            ("test", source, folds["test"]),
            ("historical", target, target.rows),
        ):
            values = {}
            for offset in TEST_OFFSETS:
                x = features(record, rows, offset)
                scores = model.decision_function(x)
                values[str(offset)] = measure(scores, targets(rows), range(len(CLASSES)))
            results[name][partition] = values
    winner = max(
        models,
        key=lambda name: np.mean(
            [value["macro_accuracy"] for value in results[name]["validation"].values()]
        ),
    )
    assert results["fixed"]["test"]["0.0"]["correct"] == 27
    assert results["fixed"]["historical"]["0.0"]["correct"] == 84
    write_json(output / "results.json", {"validation_selection": winner, "models": results})
    print(f"Dipilih pada validasi berblok: {winner}")


if __name__ == "__main__":
    main()
