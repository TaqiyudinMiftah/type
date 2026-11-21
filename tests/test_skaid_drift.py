import inspect

import numpy as np
import pytest

from ketik_audio.common import ResearchError
from ketik_audio.skaid_alignment import correlation_curve, select_peak, template
from ketik_audio.skaid_drift import (
    clock,
    fit_models,
    model_score,
    physical_events,
    weighted_clock_fit,
    window_audit,
)


def synthetic_clock(ppm):
    rng = np.random.default_rng(4248)
    stamps = np.cumsum(rng.integers(50, 150, size=4200))
    total = int(stamps[-1] // 10 + 10)
    bins, cut = stamps // 10, total // 2
    feature = rng.uniform(0, 0.001, total + 200)
    for position, amplitude in zip(
        37 + (1 + ppm / 1e6) * bins, rng.uniform(0.6, 1.4, len(bins)), strict=True
    ):
        grid = np.arange(max(0, int(position) - 5), min(len(feature), int(position) + 6))
        feature[grid] += amplitude * np.exp(-0.5 * ((grid - position) / 0.85) ** 2)
    fit = stamps[bins < cut]
    coarse = select_peak(correlation_curve(feature, template(fit // 10, cut), fit // 10))[
        "lag_bins"
    ]
    return feature, stamps, cut, total, fit_models(feature, fit, cut, coarse)


@pytest.mark.parametrize("ppm", [-120, 120])
def test_early_clock_fit_recovers_both_drift_signs_and_predicts_later_timing(ppm):
    feature, stamps, cut, total, selected = synthetic_clock(ppm)
    assert selected["affine_admitted"]
    model = selected["models"]["affine_policy"]
    assert model["drift_ppm"] == pytest.approx(ppm, abs=15)
    assert model["a_bins"] == pytest.approx(37, abs=0.3)
    late = stamps[stamps // 10 >= cut] // 10 - cut
    pattern = template(late, total - cut)
    affine = model_score(feature, pattern, late, cut, model)
    constant = model_score(feature, pattern, late, cut, selected["models"]["refined_constant"])
    assert affine["correlation"] > constant["correlation"] + 0.15
    # The fit API cannot accept late events or class labels.
    assert list(inspect.signature(fit_models).parameters) == ["feature", "fit_ms", "cut", "coarse"]
    with pytest.raises(ResearchError):
        fit_models(feature, stamps, cut, round(model["a_bins"]))


def test_rate_gate_falls_back_without_clipping_a_large_clock_drift():
    _, _, _, _, selected = synthetic_clock(350)
    assert selected["refinement_admitted"]
    assert not selected["affine_admitted"]
    assert selected["regression"]["drift_ppm"] > 200
    assert selected["models"]["affine_policy"] == selected["models"]["refined_constant"]
    assert selected["models"]["affine_policy"]["b"] == 0


def test_weighted_fit_matches_independent_least_squares_and_constant_audio_fallback():
    x = np.array([1800, 5600, 9800, 13000, 19000], dtype=float)
    y = 210.3 - 0.000125 * x + np.array([0.01, -0.02, 0.01, -0.03, 0.01])
    w = np.array([49, 62, 80, 52, 90])
    result = weighted_clock_fit(x, y, w)
    design = np.column_stack([np.ones(len(x)), x])
    independent = np.linalg.lstsq(design * np.sqrt(w[:, None]), y * np.sqrt(w), rcond=None)[0]
    assert result["a_bins"] == pytest.approx(independent[0], abs=1e-10)
    assert result["b"] == pytest.approx(independent[1], abs=1e-10)
    empty = fit_models(np.ones(5000), np.arange(0, 8000, 100), 1000, 12)
    assert not empty["affine_admitted"]
    assert empty["models"]["affine_policy"] == clock(12)
    with pytest.raises(ResearchError):
        clock(1, -1.1)


def test_window_ownership_half_open_edges_unmatched_release_and_native_bounds():
    stamps = np.array([100, 200, 205, 220, 400, 410, 430, 465, 470, 500, 990, 1020, 1100, 1150])
    kinds = [
        "press",
        "press",
        "release",
        "release",
        "press",
        "press",
        "release",
        "press",
        "release",
        "release",
        "press",
        "release",
        "press",
        "release",
    ]
    keys = ["x", "a", "x", "a", "b", "b", "b", "c", "orphan", "c", "d", "d", "e", "e"]
    phases = ["Phase 1"] * 13 + ["Phase 2"]
    presses, releases = physical_events(np.arange(2, 16), stamps, phases, kinds, keys)
    assert len(presses) == 6  # The second b press is an OS repeat while held.
    stats, arrays = window_audit(presses, releases, "Phase 1", 0, 48000, 48000, clock(0))
    assert arrays["press_key"].tolist() == ["x", "a", "b", "c", "d", "e"]
    assert arrays["w80_complete"].tolist() == [True, True, True, True, False, False]
    assert arrays["w80_no_neighbor_press"].tolist() == [True, True, True, True, False, False]
    assert arrays["w80_other_release_count"].tolist() == [0, 1, 0, 1, 0, 0]
    assert arrays["w80_no_other_key_event"].tolist() == [True, False, True, False, False, False]
    assert not arrays["same_phase_positive_release_pair"][-1]
    assert stats[-1]["tiers"]["no_other_key_event"]["events"] == 2
