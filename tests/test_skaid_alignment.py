import numpy as np
import pytest

from ketik_audio.common import ResearchError
from ketik_audio.skaid_alignment import (
    assess_timing,
    correlation_curve,
    fixed_lag_score,
    onset_feature,
    permute_gaps,
    select_peak,
    waveform_copy_metrics,
)


def test_overlap_correlations_match_direct_pearson_and_both_lag_signs():
    rng = np.random.default_rng(2409)
    pattern = rng.normal(size=77)
    audio = rng.normal(size=133)
    bins = np.array([0, 11, 23, 35, 47, 59, 76])
    curve = correlation_curve(audio, pattern, bins)
    for lag in [-76, -55, -20, 0, 24, 56, 100, 132]:
        index = np.flatnonzero(curve["lags"] == lag).item()
        direct = fixed_lag_score(audio, pattern, bins, lag)
        if direct["correlation"] is not None:
            assert curve["scores"][index] == pytest.approx(direct["correlation"], abs=1e-11)
        assert curve["event_count"][index] == direct["covered_events"]
        assert bool(curve["eligible"][index]) == direct["eligible"]
    for lag in [-17, 23]:
        audio = rng.normal(scale=0.001, size=133)
        start, end = max(-lag, 0), min(len(pattern), len(audio) - lag)
        audio[start + lag : end + lag] += pattern[start:end]
        peak = select_peak(correlation_curve(audio, pattern, bins))
        assert peak["lag_bins"] == lag
        assert peak["correlation"] > 0.999


def test_gap_controls_preserve_millisecond_gaps_and_endpoints():
    events = np.array([10, 31, 31, 50, 93, 105])
    shuffled = permute_gaps(events, np.random.default_rng(183))
    assert shuffled[[0, -1]].tolist() == events[[0, -1]].tolist()
    assert sorted(np.diff(shuffled).tolist()) == sorted(np.diff(events).tolist())
    assert np.all(np.diff(shuffled) >= 0)
    assert permute_gaps([], np.random.default_rng(2)).size == 0


def test_half_log_fit_does_not_correct_later_clock_change():
    rng = np.random.default_rng(24)
    stamps = np.cumsum(rng.integers(90, 700, size=250)).astype(np.int64)
    span = int(stamps[-1] + 100)
    base_bins = stamps // 10
    cut = (span // 10 + 1) // 2
    rms = np.ones(span // 10 + 100)
    rms[base_bins + 37] = 20
    stable, _ = assess_timing(rms, stamps, span, 17282184)
    assert stable["offset_seconds"] == pytest.approx(0.37)
    assert stable["validation_local_residual_ms"] == 0
    assert stable["validation"]["correlation"] > 0.5
    shifted_rms = np.ones_like(rms)
    shifted_rms[base_bins + 37 + np.where(base_bins >= cut, 15, 0)] = 20
    unstable, _ = assess_timing(shifted_rms, stamps, span, 17282184)
    assert unstable["offset_seconds"] == pytest.approx(0.37)
    assert unstable["validation_local_residual_ms"] == 150
    assert not unstable["provisional_stable_match"]
    assert not unstable["criteria"]["local_stability"]


def test_transformed_waveform_copy_and_unrelated_audio():
    rng = np.random.default_rng(213)
    source = rng.normal(size=41000)
    copy = np.concatenate((np.zeros(131), 0.72 * source + 0.06))
    metrics = waveform_copy_metrics(source, copy)
    assert metrics["lag_samples"] == 131
    assert metrics["least_squares_gain"] == pytest.approx(0.72)
    assert metrics["strong_transformed_copy_evidence"]
    other = waveform_copy_metrics(source, rng.normal(size=41000))
    assert not other["strong_transformed_copy_evidence"]


def test_silence_and_invalid_envelopes_do_not_produce_alignment():
    assert np.all(onset_feature(np.zeros(100)) == 0)
    curve = correlation_curve(np.ones(80), np.ones(20), np.arange(20))
    assert select_peak(curve) is None
    with pytest.raises(ResearchError):
        onset_feature(np.array([0, np.nan]))
    with pytest.raises(ResearchError):
        assess_timing(np.ones(100), np.array([30, 20]), 100, 24)
