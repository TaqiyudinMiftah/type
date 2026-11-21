"""Robust boundary-marker diagnostics; preserves the original browser audit."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from .common import LABELS, now, write_json
from .web_collection import coverage, decode_json, import_session, read_events, replay_timing
from .web_quota import CLASSES

METHOD = "calibration-affine-consensus-v2"
RESIDUAL_MS = 20.0
MAX_DRIFT = 0.005
MIN_TRAIN_PER_PHASE = 3
MIN_VALIDATION_PER_PHASE = 3
PHASES = ("sync_start", "sync_end")


def fit_affine(markers):
    if len(markers) < 2:
        return None
    x, y = np.asarray([(m["clock_frame"], m["peak_frame"]) for m in markers]).T
    xx, yy = x - x.mean(), y - y.mean()
    denominator = float(xx @ xx)
    if denominator <= 0:
        return None
    slope = float(xx @ yy / denominator)
    intercept = float(y.mean() - slope * x.mean())
    if not np.isfinite([slope, intercept]).all() or abs(slope - 1) > MAX_DRIFT:
        return None
    return {"slope": slope, "intercept": intercept}


def residual_ms(marker, affine, rate):
    return (
        (marker["peak_frame"] - affine["intercept"] - affine["slope"] * marker["clock_frame"])
        / rate
        * 1000
    )


def training_hypotheses(training, rate):
    """This function receives no validation markers or validation statistics."""
    unique = {}
    for first, second in combinations(training, 2):
        if first["phase"] == second["phase"]:
            continue
        affine = fit_affine([first, second])
        if affine is None:
            continue
        selected = [m for m in training if abs(residual_ms(m, affine, rate)) <= RESIDUAL_MS]
        # Refit a stable consensus, without consulting validation outcomes.
        for _ in range(4):
            affine = fit_affine(selected)
            if affine is None:
                break
            updated = [m for m in training if abs(residual_ms(m, affine, rate)) <= RESIDUAL_MS]
            if [m["event_seq"] for m in updated] == [m["event_seq"] for m in selected]:
                key = tuple(m["event_seq"] for m in selected)
                unique[key] = {
                    "affine": affine,
                    "training_inliers": list(key),
                    "support": len(selected),
                    "training_squared_error_ms": float(
                        sum(residual_ms(m, affine, rate) ** 2 for m in selected)
                    ),
                }
                break
            selected = updated
    return sorted(
        unique.values(),
        key=lambda h: (-h["support"], h["training_squared_error_ms"], h["training_inliers"]),
    )


def consensus_timing(legacy, rate):
    """Apply robust fitting to independently replayed, unchanged v1 peaks."""
    phases = Counter()
    rows = []
    for marker in legacy["calibration"]:
        phase = marker["phase"]
        ordinal = phases[phase]
        phases[phase] += 1
        rows.append(
            {
                **marker,
                "phase_ordinal": ordinal,
                "split": "fit" if ordinal % 2 == 0 else "validation",
            }
        )
    training = [m for m in rows if m["split"] == "fit" and m["reason"] is None]
    validation = [m for m in rows if m["split"] == "validation" and m["reason"] is None]
    hypotheses = training_hypotheses(training, rate)
    best = hypotheses[0] if hypotheses else None
    affine = best["affine"] if best else None
    inlier_ids = set(best["training_inliers"]) if best else set()
    inliers = [m for m in training if m["event_seq"] in inlier_ids]
    outliers = [m["event_seq"] for m in training if m["event_seq"] not in inlier_ids]
    train_by_phase = {p: sum(m["phase"] == p for m in inliers) for p in PHASES}
    validation_by_phase = {p: sum(m["phase"] == p for m in validation) for p in PHASES}
    residuals = (
        [
            {
                "event_seq": m["event_seq"],
                "phase": m["phase"],
                "residual_ms": residual_ms(m, affine, rate),
            }
            for m in validation
        ]
        if affine
        else []
    )
    # Only training coordinates define the ambiguity interval. Even the support
    # range must not depend on validation values or their availability.
    ambiguous = False
    if best and training:
        endpoints = [
            min(m["clock_frame"] for m in training),
            max(m["clock_frame"] for m in training),
        ]
        for alternative in hypotheses[1:]:
            if alternative["support"] != best["support"]:
                break
            other = alternative["affine"]
            separation = max(
                abs(
                    (affine["slope"] - other["slope"]) * x
                    + affine["intercept"]
                    - other["intercept"]
                )
                / rate
                * 1000
                for x in endpoints
            )
            ambiguous |= separation > RESIDUAL_MS
    issues = set(legacy["issues"]) - {"timing_residual", "uncertain_calibration"}
    if any(phases[p] != 8 for p in PHASES) or len(rows) != 16:
        issues.add("incomplete_calibration")
    if any(m["reason"] is not None for m in rows):
        issues.add("weak_or_ambiguous_calibration")
    if any(n < MIN_TRAIN_PER_PHASE for n in train_by_phase.values()):
        issues.add("insufficient_training_support")
    if any(n < MIN_VALIDATION_PER_PHASE for n in validation_by_phase.values()):
        issues.add("insufficient_validation_support")
    if affine is None:
        issues.add("robust_fit_failed")
    if ambiguous:
        issues.add("ambiguous_training_consensus")
    if any(abs(r["residual_ms"]) > RESIDUAL_MS for r in residuals):
        issues.add("validation_residual")
    for row in rows:
        if row["reason"] is not None:
            row["used_for_fit"] = False
        elif row["split"] == "fit":
            row["used_for_fit"] = row["event_seq"] in inlier_ids
        else:
            row["used_for_fit"] = False
    return {
        "method": METHOD,
        "status": "needs_review" if issues else "checked",
        "issues": sorted(issues),
        "affine": affine,
        "training_inliers": sorted(inlier_ids),
        "excluded_training_peaks": outliers,
        "training_inliers_by_phase": train_by_phase,
        "validation_markers_by_phase": validation_by_phase,
        "validation_residuals": residuals,
        "residuals_ms": [r["residual_ms"] for r in residuals],
        "max_abs_validation_residual_ms": max(
            (abs(r["residual_ms"]) for r in residuals), default=None
        ),
        "hypotheses": hypotheses,
        "ambiguous_training_consensus": bool(ambiguous),
        "calibration": rows,
        "limits": {
            "residual_ms": RESIDUAL_MS,
            "max_drift_fraction": MAX_DRIFT,
            "minimum_training_inliers_per_phase": MIN_TRAIN_PER_PHASE,
            "minimum_validation_markers_per_phase": MIN_VALIDATION_PER_PHASE,
        },
        "validation_used_for_fit_or_model_selection": False,
        "interpretation": (
            "Konsistensi penanda awal/akhir; "
            "bukan sertifikasi onset mekanik atau timing di tengah sesi."
        ),
    }


def align_package(package: Path, output: Path, evaluation_role="unclassified"):
    """Keep raw files and review.json from v1, then append a distinct v2 report."""
    import soundfile as sf

    original = import_session(package, output)
    meta = decode_json((output / "session.json").read_bytes())
    events = read_events((output / "events.jsonl").read_bytes(), meta["event_count"])
    audio, rate = sf.read(output / "audio.wav", dtype="float32")
    legacy, clock = replay_timing(
        audio,
        rate,
        meta["first_frame"],
        events,
        decode_json((output / "timing.json").read_bytes()),
        meta["capture_issues"],
    )
    result = consensus_timing(legacy, rate)
    if meta["status"] != "complete":
        result["issues"] = sorted(set(result["issues"]) | {"capture_incomplete"})
        result["status"] = "needs_review"
    result.update(
        {
            "schema": "ketik-web-alignment/v2",
            "created_at": now(),
            "source_sha256": original["source_sha256"],
            "evaluation_role": evaluation_role,
            "session_id": meta["session_id"],
            "synthetic": meta["synthetic"],
            "capture_status": meta["status"],
            "original_v1_status": original["status"],
            "original_v1_timing_status": original["timing"]["status"],
            "original_v1_max_residual_ms": max(map(abs, legacy["residuals_ms"]), default=None),
            "duration_seconds": original["duration_seconds"],
            "coverage": coverage(
                audio,
                rate,
                events,
                result,
                clock,
                CLASSES if "collection" in original else LABELS,
                original["collection"]["selected_sequences"] if "collection" in original else None,
            ),
            **({"collection": original["collection"]} if "collection" in original else {}),
            "classifier_fit_performed": False,
        }
    )
    write_json(output / "alignment-v2.json", result)
    return result
