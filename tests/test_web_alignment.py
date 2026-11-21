from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ketik_audio.web_alignment import (
    MAX_DRIFT,
    MIN_TRAIN_PER_PHASE,
    MIN_VALIDATION_PER_PHASE,
    RESIDUAL_MS,
    consensus_timing,
)
from ketik_audio.web_collection import replay_timing
from research.web_alignment_synthetic import make_case


def analyze(case, seed=7100, rate=16000):
    f = make_case(case, seed, rate)
    timing = {
        "method": "calibration-affine-energy-v1",
        "first_frame": f["first"],
        "sample_rate": rate,
        "anchors": f["anchors"],
    }
    legacy, _ = replay_timing(
        f["audio"],
        rate,
        f["first"],
        f["events"],
        timing,
        f["capture_issues"],
    )
    # The fixture is not a browser report. Bypass only this comparison flag;
    # clock, signal, ambiguity, dispatch and calibration diagnostics are real.
    legacy["issues"] = [i for i in legacy["issues"] if i != "browser_timing_mismatch"]
    legacy["status"] = "needs_review" if legacy["issues"] else "checked"
    return f, legacy, consensus_timing(legacy, rate)


@pytest.mark.parametrize("case", ["clean", "drift", "training_distractor"])
@pytest.mark.parametrize("rate", [16000, 48000])
def test_known_waveform_mapping_and_training_peak_outlier(case, rate):
    fixture, old, new = analyze(case, rate=rate)
    assert new["status"] == "checked", new["issues"]
    assert len(new["validation_residuals"]) == 8
    errors = [
        abs(
            new["affine"]["slope"] * r["clock_frame"]
            + new["affine"]["intercept"]
            - r["physical_frame"]
        )
        / rate
        * 1000
        for r in fixture["truth"]
    ]
    assert max(errors) < 10
    assert not new["validation_used_for_fit_or_model_selection"]
    if case == "training_distractor":
        assert old["status"] == "needs_review"
        assert new["excluded_training_peaks"] == [fixture["marker_seq"][0]]


@pytest.mark.parametrize(
    "case",
    [
        "validation_distractor",
        "weak_markers",
        "silence",
        "stationary_noise",
        "competing_peaks",
        "missing_end",
        "dispatch_lag",
        "capture_interrupted",
        "insufficient_training_support",
    ],
)
def test_bad_signal_or_validation_is_never_rescued_by_trimming(case):
    fixture, _, result = analyze(case)
    assert result["status"] == "needs_review"
    if case == "validation_distractor":
        assert "validation_residual" in result["issues"]
        assert len(result["validation_residuals"]) == 8
        assert fixture["marker_seq"][1] in [r["event_seq"] for r in result["validation_residuals"]]


def test_validation_values_cannot_change_the_selected_training_fit():
    _, original, before = analyze("clean")
    changed = copy.deepcopy(original)
    for i, marker in enumerate(changed["calibration"]):
        if i % 2:
            marker["clock_frame"] += 1250
            marker["peak_frame"] += 5000
    after = consensus_timing(changed, 16000)
    assert after["affine"] == before["affine"]
    assert after["hypotheses"] == before["hypotheses"]
    assert after["training_inliers"] == before["training_inliers"]
    assert "validation_residual" in after["issues"]


@pytest.mark.parametrize("seed", [7103, 7108])
def test_comparable_distractor_peaks_remain_ambiguous(seed):
    _, _, result = analyze("training_competing_distractor", seed=seed, rate=48000)
    assert result["status"] == "needs_review"
    assert "weak_or_ambiguous_calibration" in result["issues"]


def test_equal_training_support_with_different_offsets_stays_ambiguous():
    _, original, _ = analyze("clean")
    # Only two conflicting early fit peaks and two matching late fit peaks.
    for i in [2, 4, 8, 14]:
        original["calibration"][i] = {
            "event_seq": original["calibration"][i]["event_seq"],
            "phase": original["calibration"][i]["phase"],
            "reason": "weak_signal",
        }
    original["calibration"][0]["peak_frame"] -= 16000 * 0.080
    result = consensus_timing(original, 16000)
    assert "ambiguous_training_consensus" in result["issues"]
    assert "insufficient_training_support" in result["issues"]
    assert result["status"] == "needs_review"


def test_boundary_calibration_does_not_establish_interior_timing():
    fixture, _, result = analyze("hidden_middle_bend")
    assert result["status"] == "checked"
    interior = [r for r in fixture["truth"] if r["phase"] == "typing"]
    errors = [
        abs(
            result["affine"]["slope"] * r["clock_frame"]
            + result["affine"]["intercept"]
            - r["physical_frame"]
        )
        / fixture["rate"]
        * 1000
        for r in interior
    ]
    assert max(errors) > 60  # Explicitly document the boundary-only blind spot.


def test_protocol_matches_implemented_bounds():
    protocol = json.loads(
        (Path(__file__).parents[1] / "research/WEB_ALIGNMENT_V2_PROTOCOL.json").read_text()
    )
    limits = protocol["limits"]
    assert limits["residual_ms"] == RESIDUAL_MS
    assert limits["max_drift_fraction"] == MAX_DRIFT
    assert limits["minimum_training_inliers_per_phase"] == MIN_TRAIN_PER_PHASE
    assert limits["minimum_validation_markers_per_phase"] == MIN_VALIDATION_PER_PHASE
