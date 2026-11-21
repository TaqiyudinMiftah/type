"""Timestamp-conditioned transfer of the saved Lenovo LOQ classifiers."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.base import clone
from sklearn.metrics import f1_score

from ketik_audio.audio import extract
from ketik_audio.common import FeatureConfig, file_hash, fresh_directory, read_json, write_json
from ketik_audio.skaid_pilot_features import mfcc_features
from ketik_audio.web_alignment import consensus_timing
from ketik_audio.web_collection import letter_event, read_events, replay_timing, validate_metadata
from ketik_audio.web_quota import CLASSES, SCHEMA, event_class, selection

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/lenovo-loq-calibration-f558fd22"
SOURCE = ROOT / "data/collected/lenovo-loq-calibration-f558fd22"
HISTORICAL = ROOT / "data/collected/web-calibration-2026-11-20-e9be52b8"
PROTOCOL = ROOT / "research/LOQ_TRANSFER_PROTOCOL.md"
BUDGETS = (1, 2, 3, 5, 10)
SEEDS = tuple(range(200, 210))
RATE = 48000
GUARD = round(0.150 * RATE)


@dataclass
class Recording:
    meta: dict
    audio: np.ndarray
    rows: list[dict]
    timing: dict
    typing_range: tuple[float, float]


def metadata(directory):
    meta = read_json(directory / "session.json")
    validate_metadata(meta)
    for file, expected in meta["files"].items():
        if file_hash(directory / file) != expected:
            raise ValueError(f"Hash paket berubah: {file}")
    if meta["status"] != "complete" or meta["synthetic"] or meta["capture_issues"]:
        raise ValueError("Rekaman harus nyata, lengkap, dan tanpa gangguan capture.")
    return meta


def timing_and_energy(audio, events, clock):
    size = round(0.005 * RATE)
    bins = audio[: len(audio) // size * size].reshape(-1, size)
    energy = bins.std(axis=1, dtype=np.float64)
    centers = (np.arange(len(energy)) + 0.5) * size
    quiet = np.sort(energy[100:500])
    threshold = float(max(1e-6, np.median(quiet) * 10**0.3, 1.2 * quiet[int(0.95 * 399)]))
    markers = []
    for event in events:
        if not event.get("calibration"):
            continue
        row = {"event_seq": event["seq"], "phase": event["phase"]}
        at = clock(event["time_ms"])
        if at is None:
            markers.append({**row, "reason": "outside_clock"})
            continue
        lo, hi = np.searchsorted(centers, [at - 0.065 * RATE, at + 0.065 * RATE])
        if hi <= lo or energy[lo:hi].max() <= threshold:
            markers.append({**row, "reason": "weak_signal"})
            continue
        peak = lo + int(energy[lo:hi].argmax())
        distant = np.abs(centers[lo:hi] - centers[peak]) > 0.025 * RATE
        competing = np.any(energy[lo:hi][distant] >= 0.9 * energy[peak])
        markers.append(
            {
                **row,
                "clock_frame": at,
                "peak_frame": float(centers[peak]),
                "reason": "ambiguous_peak" if competing else None,
            }
        )
    audit = consensus_timing({"calibration": markers, "issues": []}, RATE)
    audit["method"] = "pilot-ac-rms-5ms-local-peak-affine"
    return audit, centers, energy, threshold


def permits_exploration(audit):
    return (
        audit["affine"] is not None
        and set(audit["issues"])
        <= {"insufficient_training_support", "weak_or_ambiguous_calibration"}
        and min(audit["training_inliers_by_phase"].values()) >= 2
        and min(audit["validation_markers_by_phase"].values()) >= 3
        and audit["max_abs_validation_residual_ms"] is not None
        and audit["max_abs_validation_residual_ms"] <= 20
    )


def recording(directory, meta, out, partition, exploratory=False):
    audio, rate = sf.read(directory / "audio.wav", dtype="float32")
    if rate != RATE or audio.ndim != 1 or len(audio) != meta["frames"]:
        raise ValueError("WAV harus mono 48 kHz dan cocok dengan frame metadata.")
    events = read_events((directory / "events.jsonl").read_bytes(), meta["event_count"])
    standard, clock = replay_timing(
        audio,
        rate,
        meta["first_frame"],
        events,
        read_json(directory / "timing.json"),
        meta["capture_issues"],
    )
    audit, centers, energy, threshold = timing_and_energy(audio, events, clock)
    audit["standard_status"] = standard["status"]
    audit["standard_issues"] = standard["issues"]
    write_json(out / f"{partition}-timing.json", audit)
    if audit["status"] != "checked" and not (exploratory and permits_exploration(audit)):
        raise ValueError(f"Audit timing {partition} gagal: {audit['issues']}")

    affine = audit["affine"]
    candidate = [event for event in events if letter_event(event, CLASSES)]
    if meta["schema"] == SCHEMA:
        calculated = selection(events)
        if calculated != meta["collection"] or not calculated["complete"]:
            raise ValueError("Pemilihan kuota tidak cocok dengan log.")
        selected = set(calculated["selected_sequences"])
        candidate = [event for event in candidate if event["seq"] in selected]
    clock_frames = {}
    for event in events:
        if event["type"] in {"keydown", "keyup"} and not event["repeat"]:
            value = clock(event["time_ms"])
            if value is not None:
                clock_frames[event["seq"]] = affine["intercept"] + affine["slope"] * value
    nearby = np.asarray(sorted(clock_frames.values()))
    rows, rejects = [], Counter()
    for event in candidate:
        seq = event["seq"]
        at = clock_frames.get(seq)
        if at is None:
            rejects["outside_clock"] += 1
            continue
        start, end = round(at - 0.015 * rate), round(at + 0.065 * rate)
        lo, hi = np.searchsorted(nearby, [start, end])
        left, right = np.searchsorted(centers, [at - 0.015 * rate, at + 0.035 * rate])
        if start < 0 or end > len(audio):
            rejects["boundary"] += 1
        elif hi - lo > 1:
            rejects["neighbor_press_or_release"] += 1
        elif not np.isfinite(audio[start:end]).all() or np.max(np.abs(audio[start:end])) >= 0.999:
            rejects["clipping_or_invalid"] += 1
        elif right <= left or energy[left:right].max() <= threshold:
            rejects["weak_local_transient"] += 1
        else:
            rows.append(
                {
                    "event_seq": seq,
                    "key": event_class(event),
                    "frame": at,
                    "start": start,
                    "end": end,
                }
            )
    counts = Counter(row["key"] for row in rows)
    write_json(
        out / f"{partition}-coverage.json",
        {
            "eligible": len(candidate),
            "used": len(rows),
            "counts": dict(counts),
            "excluded": dict(rejects),
            "wav_sha256": meta["files"]["audio.wav"],
        },
    )
    return Recording(
        meta,
        audio,
        rows,
        audit,
        (
            min(clock_frames[e["seq"]] for e in candidate if e["seq"] in clock_frames),
            max(clock_frames[e["seq"]] for e in candidate if e["seq"] in clock_frames),
        ),
    )


def microphone(meta):
    track = meta["track_settings"]
    return track["microphone_label"], track["device_id_hash"]


def check_pair(source, support, query, condition):
    metas = (source, support, query)
    if any(meta["schema"] != SCHEMA for meta in metas):
        raise ValueError("Protokol prospektif memakai sesi kuota v2.")
    for name in ("session_id", "participant_id"):
        cardinality = len({meta[name] for meta in metas})
        if cardinality != (3 if name == "session_id" else 1):
            raise ValueError(f"Identitas sesi atau peserta tidak sesuai: {name}")
    for file in ("audio.wav", "events.jsonl"):
        if len({meta["files"][file] for meta in metas}) != 3:
            raise ValueError(f"Peran memakai ulang berkas: {file}")
    if not support["local_day"] < query["local_day"]:
        raise ValueError("Sesi query harus di hari setelah sesi dukungan.")
    if len({meta["study"]["setup"]["laptop"].strip().casefold() for meta in metas}) != 1:
        raise ValueError("Alias laptop harus cocok antara ketiga paket.")
    if source["role"] != "calibration" or microphone(support) != microphone(query):
        raise ValueError("Dukungan dan query harus memakai mikrofon target yang sama.")
    if condition == "same-mic":
        eligible = (
            source["study_id"] == support["study_id"] == query["study_id"]
            and source["local_day"] < support["local_day"]
            and microphone(source) == microphone(query)
            and (support["role"], query["role"]) == ("test1", "test2")
        )
    else:
        eligible = (
            source["study_id"] != support["study_id"] == query["study_id"]
            and microphone(source) != microphone(query)
            and (support["role"], query["role"]) == ("calibration", "test1")
        )
    if not eligible:
        raise ValueError(f"Paket tidak cocok dengan kondisi {condition} pada protokol.")


def waves(record, rows, shift_frames=0):
    segments = [
        record.audio[row["start"] + shift_frames : row["end"] + shift_frames] for row in rows
    ]
    if not segments or any(len(segment) != 3840 for segment in segments):
        raise ValueError("Potongan melampaui WAV atau panjang bukan 80 ms.")
    return resample_poly(np.stack(segments), up=1, down=3, axis=1).astype(np.float32)


def targets(rows):
    return np.array([CLASSES.index(row["key"]) for row in rows])


def measure(scores, actual, classes):
    order = np.argsort(-scores, axis=1, kind="stable")
    classes = np.asarray(classes)
    predicted = classes[order[:, 0]]
    correct = predicted == actual
    per_key = [np.mean(correct[actual == key]) for key in np.unique(actual)]
    return {
        "n": len(actual),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "macro_accuracy": float(np.mean(per_key)),
        "macro_f1": float(
            f1_score(actual, predicted, labels=np.unique(actual), average="macro", zero_division=0)
        ),
        "top3": float(np.mean(np.any(classes[order[:, :3]] == actual[:, None], axis=1))),
    }


def saved_models(wave, mfcc, labels, config, checkpoints):
    import torch

    from ketik_audio.cnn_repair import repaired_model

    torch.set_num_threads(4)
    expected = checkpoints["artifacts"]
    models = RUN / "models"
    cnn_name = f"resnet-seed-{checkpoints['models']['logmel-resnet']['selected_seed']}.pt"
    for name in ("mfcc-ridge.joblib", "mfcc-rbf.joblib", cnn_name):
        if file_hash(models / name) != expected[name]:
            raise ValueError(f"Model sumber berubah: {name}")
    ridge = joblib.load(models / "mfcc-ridge.joblib")
    svm = joblib.load(models / "mfcc-rbf.joblib")
    checkpoint = torch.load(models / cnn_name, map_location="cpu", weights_only=True)
    model, _ = repaired_model(
        checkpoints["models"]["logmel-resnet"]["selected_seed"], "compact-resnet"
    )
    model[-1] = torch.nn.Linear(model[-1].in_features, len(CLASSES))
    model.load_state_dict(checkpoint["state_dict"])
    mel = np.stack([extract(segment, config)[0] for segment in wave])
    mean, std = [
        np.asarray(checkpoint[name], dtype=np.float32)[None, :, None] for name in ("mean", "std")
    ]
    x = torch.from_numpy(((mel - mean) / std)[:, None])
    model.eval()
    with torch.inference_mode():
        logits = torch.cat([model(batch) for batch in x.split(32)]).numpy()
    scores = {
        "ridge": ridge.decision_function(mfcc),
        "svm": svm.decision_function(mfcc),
        "cnn": logits,
    }
    letters = labels != CLASSES.index("space")
    results = {}
    for name, value in scores.items():
        metrics = measure(value, labels, range(len(CLASSES)))
        if letters.any():
            metrics["letters_only"] = measure(value[letters], labels[letters], range(len(CLASSES)))
        results[name] = metrics
    return ridge, results


def budget_curve(
    ridge, source_x, source_y, support_x, support_y, query_x, query_y, classes, budgets
):
    use_source = np.isin(source_y, classes)
    base_x, base_y = source_x[use_source], source_y[use_source]
    baseline = clone(ridge).fit(base_x, base_y)
    zero = measure(baseline.decision_function(query_x), query_y, baseline.classes_)
    runs = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        orders = [rng.permutation(np.flatnonzero(support_y == key)) for key in classes]
        for k in budgets:
            take = np.concatenate([order[:k] for order in orders])
            x, y = support_x[take], support_y[take]
            variants = {
                "source_and_target": (
                    np.concatenate([base_x, x]),
                    np.concatenate([base_y, y]),
                    len(base_y),
                ),
                "target_only": (x, y, 0),
                "shuffled_target": (x, rng.permutation(y), 0),
            }
            for variant, (fit_x, fit_y, source_n) in variants.items():
                model = clone(ridge).fit(fit_x, fit_y)
                row = measure(model.decision_function(query_x), query_y, model.classes_)
                runs.append(
                    {
                        "method": variant,
                        "k": k,
                        "seed": seed,
                        "source_labels": source_n,
                        "target_labels": len(y),
                        "support_indices": take.tolist(),
                        **row,
                    }
                )
    return {"baseline": {"source_labels": len(base_y), "target_labels": 0, **zero}, "runs": runs}


def select_cohort(by_support, by_query, historical):
    if historical:
        classes = [key for key in CLASSES if by_support[key] >= 10 and by_query[key] >= 5]
        return (classes, BUDGETS) if len(classes) >= 3 else ([], ())
    if all(by_query[key] >= 5 for key in CLASSES):
        budgets = tuple(k for k in BUDGETS if all(by_support[key] >= k for key in CLASSES))
        return (list(CLASSES), budgets) if budgets else ([], ())
    return [], ()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("historical", "prospective"))
    parser.add_argument("--support", type=Path)
    parser.add_argument("--query", type=Path)
    parser.add_argument("--condition", choices=("same-mic", "new-mic"), default="same-mic")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    historical = args.mode == "historical"
    if historical and any((args.support, args.query, args.condition != "same-mic")):
        parser.error("Pilot historis dipatok ke e9be52b8.")
    if not historical and not (args.support and args.query):
        parser.error("Evaluasi prospektif perlu dua direktori impor, --support dan --query.")
    source = metadata(SOURCE)
    query_path = HISTORICAL if historical else args.query
    query_meta = metadata(query_path)
    if historical:
        if (
            not query_meta["session_id"].startswith("e9be52b8")
            or microphone(source) != microphone(query_meta)
            or source["participant_id"] != query_meta["participant_id"]
        ):
            raise ValueError("Identitas rekaman historis berubah.")
    else:
        support_meta = metadata(args.support)
        check_pair(source, support_meta, query_meta, args.condition)

    out = args.output.resolve()
    fresh_directory(out)
    query = recording(query_path, query_meta, out, "query", exploratory=historical)
    if historical:
        first, last = query.typing_range
        split = first + (last - first) * (2 / 3)
        support, support_rows = query, [row for row in query.rows if row["frame"] < split - GUARD]
        query_rows = [row for row in query.rows if row["frame"] > split + GUARD]
        assert all(a["end"] <= b["start"] for a in support_rows[-1:] for b in query_rows[:1])
    else:
        support = recording(args.support, support_meta, out, "support")
        support_rows, query_rows = support.rows, query.rows
        split = None
    by_support, by_query = [
        Counter(row["key"] for row in group) for group in (support_rows, query_rows)
    ]
    cohort, budgets = select_cohort(by_support, by_query, historical)
    write_json(
        out / "plan.json",
        {
            "mode": args.mode,
            "condition": args.condition,
            "protocol_sha256": file_hash(PROTOCOL),
            "script_sha256": file_hash(Path(__file__)),
            "source": source["session_id"],
            "support": support.meta["session_id"],
            "query": query.meta["session_id"],
            "checksums": [meta["files"] for meta in (source, support.meta, query.meta)],
            "support_counts": dict(by_support),
            "query_counts": dict(by_query),
            "classes": cohort,
            "available_budgets": list(budgets),
            "cutoff_s": split / RATE if split is not None else None,
        },
    )
    support_rows, query_rows = [
        [row for row in group if row["key"] in cohort] for group in (support_rows, query_rows)
    ]
    membership_rows = [
        (label, row)
        for label, rows in (("support", support_rows), ("query", query_rows))
        for row in rows
    ]
    if membership_rows:
        with (out / "membership.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["role", *query_rows[0]])
            writer.writeheader()
            writer.writerows({"role": label, **row} for label, row in membership_rows)

    checkpoints = read_json(RUN / "models/results.json")
    prepared = RUN / "prepared"
    if file_hash(prepared / "features.npz") != checkpoints["features_sha256"]:
        raise ValueError("Dataset fitur sumber berubah.")
    if file_hash(prepared / "dataset.json") != checkpoints["dataset_sha256"]:
        raise ValueError("Manifest dataset sumber berubah.")
    config = FeatureConfig.from_dict(read_json(prepared / "dataset.json")["cnn_features"])
    fixed_wave = waves(query, query.rows)
    full_x, full_y = mfcc_features(fixed_wave), targets(query.rows)
    ridge, zero = saved_models(fixed_wave, full_x, full_y, config, checkpoints)
    curve = None
    if budgets:
        index = {row["event_seq"]: i for i, row in enumerate(query.rows)}
        query_x = full_x[[index[row["event_seq"]] for row in query_rows]]
        with np.load(prepared / "features.npz", allow_pickle=False) as z:
            train = z["fold"] == "train"
            curve = budget_curve(
                ridge,
                z["mfcc"][train],
                z["labels"][train],
                mfcc_features(waves(support, support_rows)),
                targets(support_rows),
                query_x,
                targets(query_rows),
                [CLASSES.index(k) for k in cohort],
                budgets,
            )
    sensitivity = {}
    if historical:
        for shift_ms in (-12.5, 12.5):
            shifted = mfcc_features(waves(query, query.rows, round(shift_ms * RATE / 1000)))
            sensitivity[str(shift_ms)] = measure(
                ridge.decision_function(shifted), full_y, ridge.classes_
            )
    write_json(
        out / "results.json",
        {
            "mode": args.mode,
            "condition": args.condition,
            "query_timing": query.timing["status"],
            "hours_apart": abs(
                (
                    datetime.fromisoformat(source["started_at"])
                    - datetime.fromisoformat(query.meta["started_at"])
                ).total_seconds()
            )
            / 3600,
            "full_k0": zero,
            "ridge_sensitivity_ms": sensitivity,
            "curve": curve,
            "uniform_chance_full": 1 / len(CLASSES),
            "uniform_chance_cohort": 1 / len(cohort) if cohort else None,
            "available_budgets": list(budgets),
            "query_class_counts": dict(Counter(row["key"] for row in query.rows)),
            "cohort": cohort,
            "cohort_query_count": len(query_rows),
            "hardware": "CPU",
        },
    )
    print(f"{args.mode}: {len(query.rows)} potongan K0, {len(cohort)} kelas kurva, K={budgets}.")


if __name__ == "__main__":
    main()
