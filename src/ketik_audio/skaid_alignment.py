"""Timing-only diagnostics; positive lag places a log event later in the audio."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate, correlation_lags, find_peaks

from .common import ResearchError

BIN_SECONDS = 0.01


def onset_feature(rms):
    rms = np.asarray(rms, dtype=np.float64)
    if rms.ndim != 1 or not len(rms) or np.any(rms < 0) or not np.isfinite(rms).all():
        raise ResearchError("Envelope RMS tidak valid.")
    floor = max(float(np.median(rms)) * 0.001, 1e-12)
    log = np.log(np.maximum(rms, floor))
    feature = np.maximum(np.diff(log, prepend=log[0]), 0)
    return np.minimum(feature, np.percentile(feature, 99.5))


def template(event_bins, length):
    bins = np.asarray(event_bins, dtype=np.int64)
    if length <= 0 or np.any(bins < 0) or np.any(bins >= length):
        raise ResearchError("Bin event/template di luar batas.")
    impulses = np.bincount(bins, minlength=length).astype(np.float64)
    return gaussian_filter1d(impulses, 1.0, mode="constant", truncate=3.0)


def _moments(values, starts, ends):
    sums = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    squares = np.concatenate(([0.0], np.cumsum(values * values, dtype=np.float64)))
    return sums[ends] - sums[starts], squares[ends] - squares[starts]


def correlation_curve(audio, pattern, event_bins):
    """Pearson per overlap, with explicit geometry and event coverage masks."""
    audio, pattern = np.asarray(audio, dtype=np.float64), np.asarray(pattern, dtype=np.float64)
    if (
        not len(audio)
        or not len(pattern)
        or not np.isfinite(audio).all()
        or not np.isfinite(pattern).all()
    ):
        raise ResearchError("Input korelasi kosong/tidak finite.")
    bins = np.asarray(event_bins, dtype=np.int64)
    if np.any(bins < 0) or np.any(bins >= len(pattern)):
        raise ResearchError("Event di luar pattern.")
    lags = correlation_lags(len(audio), len(pattern), mode="full")
    a_start, a_end = np.maximum(lags, 0), np.minimum(len(audio), lags + len(pattern))
    p_start, p_end = np.maximum(-lags, 0), np.minimum(len(pattern), len(audio) - lags)
    n = a_end - a_start
    a_sum, a_sq = _moments(audio, a_start, a_end)
    p_sum, p_sq = _moments(pattern, p_start, p_end)
    numerator = correlate(audio, pattern, mode="full", method="fft") - a_sum * p_sum / n
    denominator = np.sqrt(
        np.maximum(a_sq - a_sum * a_sum / n, 0) * np.maximum(p_sq - p_sum * p_sum / n, 0)
    )
    scores = np.full(len(lags), np.nan)
    good = denominator > 1e-10
    scores[good] = np.clip(numerator[good] / denominator[good], -1, 1)
    counts = np.concatenate(([0], np.cumsum(np.bincount(bins, minlength=len(pattern)))))
    event_count = counts[p_end] - counts[p_start]
    eligible = good & (2 * n >= len(pattern)) & (2 * event_count >= len(bins))
    if not len(bins):
        eligible[:] = False
    return {"lags": lags, "scores": scores, "eligible": eligible, "event_count": event_count}


def select_peak(curve, low=None, high=None):
    eligible = curve["eligible"].copy()
    if low is not None:
        eligible &= curve["lags"] >= low
    if high is not None:
        eligible &= curve["lags"] <= high
    if not eligible.any():
        return None
    index = int(np.argmax(np.where(eligible, curve["scores"], -np.inf)))
    return {
        "index": index,
        "lag_bins": int(curve["lags"][index]),
        "correlation": float(curve["scores"][index]),
        "covered_events": int(curve["event_count"][index]),
    }


def fixed_lag_score(audio, pattern, event_bins, lag):
    start, end = max(-lag, 0), min(len(pattern), len(audio) - lag)
    if end <= start:
        return {"correlation": None, "covered_events": 0, "coverage": 0.0, "eligible": False}
    a = audio[start + lag : end + lag]
    p = pattern[start:end]
    a, p = a - np.mean(a), p - np.mean(p)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(p))
    count = int(((event_bins >= start) & (event_bins < end)).sum())
    score = float(np.clip(np.dot(a, p) / denominator, -1, 1)) if denominator > 1e-10 else None
    return {
        "correlation": score,
        "covered_events": count,
        "coverage": count / len(event_bins) if len(event_bins) else 0.0,
        "eligible": bool(
            score is not None
            and 2 * (end - start) >= len(pattern)
            and 2 * count >= len(event_bins)
            and len(event_bins)
        ),
    }


def permute_gaps(event_bins, rng):
    """Permute exact 1 ms gaps outside this helper when binning is needed."""
    values = np.asarray(event_bins, dtype=np.int64)
    if not len(values):
        return values.copy()
    return np.concatenate((values[:1], values[0] + np.cumsum(rng.permutation(np.diff(values)))))


def waveform_copy_metrics(reference, candidate, rate=16000):
    """Check only shifts within ±1 second; do not choose a transform family."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    x, y = reference - reference.mean(), candidate - candidate.mean()
    cross = correlate(y, x, mode="full", method="fft")
    lags = correlation_lags(len(y), len(x), mode="full")
    allowed = np.abs(lags) <= rate
    index = int(np.argmax(np.where(allowed, cross, -np.inf)))
    lag = int(lags[index])
    start, end = max(-lag, 0), min(len(x), len(y) - lag)
    x, y = x[start:end], y[start + lag : end + lag]
    x, y = x - x.mean(), y - y.mean()
    gain = float(np.dot(x, y) / np.dot(x, x))
    residual = y - gain * x
    relative = float(np.linalg.norm(residual) / np.linalg.norm(y))
    pearson = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    return {
        "lag_samples": lag,
        "lag_seconds": lag / rate,
        "overlap_samples": len(x),
        "pearson": pearson,
        "least_squares_gain": gain,
        "residual_rms": float(np.sqrt(np.mean(residual * residual))),
        "relative_residual_rms": relative,
        "strong_transformed_copy_evidence": pearson >= 0.999 and relative <= 0.05,
    }


def assess_timing(rms, relative_ms, span_ms, seed):
    """Fit a constant lag; later timing templates are diagnostics, never corrections."""
    relative_ms = np.asarray(relative_ms, dtype=np.int64)
    if (
        span_ms <= 0
        or np.any(relative_ms < 0)
        or np.any(relative_ms > span_ms)
        or np.any(np.diff(relative_ms) < 0)
    ):
        raise ResearchError("Rentang/urutan waktu press tidak valid.")
    feature = onset_feature(rms)
    length = span_ms // 10 + 1
    cut = length // 2
    all_bins = relative_ms // 10
    fit_ms = relative_ms[all_bins < cut]
    val_ms = relative_ms[all_bins >= cut]
    fit_bins, val_bins = fit_ms // 10, val_ms // 10 - cut
    fit_pattern, val_pattern = template(fit_bins, cut), template(val_bins, length - cut)
    curve = correlation_curve(feature, fit_pattern, fit_bins)
    peak = select_peak(curve)
    arrays = {
        "onset": feature,
        "relative_press_ms": relative_ms,
        "fit_template": fit_pattern,
        "validation_template": val_pattern,
        **{f"fit_{k}": v for k, v in curve.items()},
    }
    result = {
        "seed": seed,
        "initial_physical_presses": len(relative_ms),
        "span_ms": int(span_ms),
        "bin_seconds": BIN_SECONDS,
        "fit_bins": cut,
        "total_bins": length,
        "fit_events": len(fit_ms),
        "validation_events": len(val_ms),
        "fit_peak": peak,
        "provisional_stable_match": False,
    }
    if peak is None:
        result["reason"] = "no eligible finite fit correlation"
        return result, arrays
    lag = peak["lag_bins"]
    alternative_curve = {
        **curve,
        "eligible": curve["eligible"] & (np.abs(curve["lags"] - lag) > 25),
    }
    alternative = select_peak(alternative_curve)
    margin = peak["correlation"] - alternative["correlation"] if alternative else None
    validation = fixed_lag_score(feature, val_pattern, val_bins, lag + cut)
    val_curve = correlation_curve(feature, val_pattern, val_bins)
    val_local = select_peak(val_curve, low=lag + cut - 25, high=lag + cut + 25)
    val_delta = val_local["lag_bins"] - (lag + cut) if val_local else None
    rng = np.random.default_rng(seed)
    fit_controls = []
    for _ in range(19):
        bins = permute_gaps(fit_ms, rng) // 10
        control_peak = select_peak(correlation_curve(feature, template(bins, cut), bins))
        fit_controls.append(control_peak["correlation"] if control_peak else np.nan)
    val_controls = []
    for _ in range(99):
        bins = permute_gaps(val_ms, rng) // 10 - cut
        control = fixed_lag_score(feature, template(bins, length - cut), bins, lag + cut)
        val_controls.append(
            control["correlation"] if control["correlation"] is not None else np.nan
        )
    arrays["fit_null_max_correlation"] = np.asarray(fit_controls)
    arrays["validation_null_fixed_correlation"] = np.asarray(val_controls)
    fit_null_valid, val_null_valid = np.isfinite(fit_controls), np.isfinite(val_controls)
    fit_tail = (1 + int(np.sum(np.asarray(fit_controls) >= peak["correlation"]))) / 20
    val_score = validation["correlation"]
    val_tail = (
        (1 + int(np.sum(np.asarray(val_controls) >= val_score))) / 100
        if val_score is not None
        else None
    )
    quarters = []
    for q in range(4):
        start, end = length * q // 4, length * (q + 1) // 4
        bins = all_bins[(all_bins >= start) & (all_bins < end)] - start
        pattern = template(bins, end - start)
        local_curve = correlation_curve(feature, pattern, bins)
        local = select_peak(local_curve, low=lag + start - 25, high=lag + start + 25)
        quarters.append(
            {
                "quarter": q + 1,
                "events": len(bins),
                "fixed_offset": fixed_lag_score(feature, pattern, bins, lag + start),
                "local_best": local,
                "residual_lag_ms": (local["lag_bins"] - lag - start) * 10 if local else None,
            }
        )
    positions = all_bins + lag
    in_audio = (positions >= 0) & (positions < len(feature))
    selected_val = positions[(all_bins >= cut) & in_audio]
    onsets = find_peaks(feature, height=np.percentile(feature, 75), distance=3)[0]
    distance_stats = None
    if len(onsets) and len(selected_val):
        indices = np.searchsorted(onsets, selected_val)
        left, right = (
            onsets[np.clip(indices - 1, 0, len(onsets) - 1)],
            onsets[np.clip(indices, 0, len(onsets) - 1)],
        )
        distances_ms = np.minimum(np.abs(selected_val - left), np.abs(selected_val - right)) * 10
        arrays["validation_nearest_onset_distance_ms"] = distances_ms
        distance_stats = {
            "events": len(selected_val),
            "median_ms": float(np.median(distances_ms)),
            "p90_ms": float(np.percentile(distances_ms, 90)),
            "fraction_within_30ms": float(np.mean(distances_ms <= 30)),
        }
    arrays["press_in_audio"] = in_audio
    fit_coverage = peak["covered_events"] / len(fit_ms) if len(fit_ms) else 0
    flags = {
        "enough_events": len(fit_ms) >= 25 and len(val_ms) >= 25,
        "coverage": fit_coverage >= 0.8 and validation["coverage"] >= 0.8,
        "correlation": peak["correlation"] >= 0.10 and val_score is not None and val_score >= 0.10,
        "unique_peak": margin is not None and margin >= 0.02,
        "fit_controls": bool(fit_null_valid.all() and fit_tail == 0.05),
        "validation_controls": bool(val_null_valid.all() and val_tail == 0.01),
        "local_stability": val_delta is not None and abs(val_delta) <= 3,
    }
    result.update(
        {
            "offset_seconds": lag * BIN_SECONDS,
            "fit_event_coverage": fit_coverage,
            "alternative_peak": alternative,
            "fit_peak_margin": margin,
            "fit_control_tail_fraction": fit_tail,
            "fit_control_finite": int(fit_null_valid.sum()),
            "validation": validation,
            "validation_local_best": val_local,
            "validation_local_residual_ms": val_delta * 10 if val_delta is not None else None,
            "validation_control_tail_fraction": val_tail,
            "validation_control_finite": int(val_null_valid.sum()),
            "quarters": quarters,
            "press_in_audio": int(in_audio.sum()),
            "press_before_audio": int((positions < 0).sum()),
            "press_after_audio": int((positions >= len(feature)).sum()),
            "validation_nearest_onset": distance_stats,
            "criteria": flags,
            "provisional_stable_match": all(flags.values()),
        }
    )
    return result, arrays
