"""Fit clock drift from early event timing, then diagnose fixed models and crop overlap."""

from __future__ import annotations

import numpy as np

from .common import ResearchError
from .skaid_alignment import correlation_curve, select_peak, template

METHODS = ("legacy", "refined_constant", "affine_policy")
WINDOWS = ((400, 50), (200, 30), (120, 20), (80, 15))


def clock(a, b=0.0, source="legacy"):
    if not np.isfinite(a) or not np.isfinite(b) or 1 + b <= 0:
        raise ResearchError("Model jam tidak finite/monotonik.")
    return {"a_bins": float(a), "b": float(b), "drift_ppm": float(b * 1e6), "source": source}


def weighted_clock_fit(x, y, weights):
    x, y, w = (np.asarray(v, dtype=np.float64) for v in (x, y, weights))
    if len(x) < 2 or len(x) != len(y) or len(x) != len(w) or np.any(w <= 0):
        raise ResearchError("Blok regresi tidak valid.")
    if not all(np.isfinite(v).all() for v in (x, y, w)):
        raise ResearchError("Regresi tidak finite.")
    x_mean, y_mean = np.average(x, weights=w), np.average(y, weights=w)
    denominator = np.sum(w * (x - x_mean) ** 2)
    if denominator <= 0:
        raise ResearchError("Regresi waktu tidak memiliki rentang.")
    slope = float(np.sum(w * (x - x_mean) * (y - y_mean)) / denominator)
    intercept = float(y_mean - slope * x_mean)
    residual = y - intercept - slope * x
    return {
        "constant_a_bins": float(y_mean),
        "a_bins": intercept,
        "b": slope,
        "drift_ppm": slope * 1e6,
        "max_abs_residual_ms": float(np.max(np.abs(residual)) * 10),
        "residual_bins": residual.tolist(),
        "x_span_bins": float(np.ptp(x)),
    }


def block_offset(feature, fit_bins, start, end, coarse):
    event_bins = fit_bins[(fit_bins >= start) & (fit_bins < end)] - start
    expected = coarse + start
    lo, hi = max(0, expected - 10), min(len(feature), expected + 10 + end - start)
    result = {
        "start_bin": int(start),
        "end_bin": int(end),
        "events": len(event_bins),
        "eligible": False,
        "peak": None,
        "alternative": None,
    }
    if hi <= lo or not len(event_bins):
        return result
    curve = correlation_curve(feature[lo:hi], template(event_bins, end - start), event_bins)
    allowed = (curve["lags"] >= expected - 10 - lo) & (curve["lags"] <= expected + 10 - lo)
    curve["eligible"] &= allowed
    peak = select_peak(curve)
    if peak is None:
        return result
    alternative = select_peak(
        {**curve, "eligible": curve["eligible"] & (np.abs(curve["lags"] - peak["lag_bins"]) > 3)}
    )
    margin = peak["correlation"] - alternative["correlation"] if alternative else None
    i, fraction = peak["index"], 0.0
    if i > 0 and i + 1 < len(curve["scores"]) and curve["eligible"][[i - 1, i + 1]].all():
        left, center, right = curve["scores"][[i - 1, i, i + 1]]
        curvature = left - 2 * center + right
        if curvature < 0:
            fraction = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
    coarse_local = peak["lag_bins"] + lo
    mapped = event_bins + coarse_local
    covered = (mapped >= 0) & (mapped < len(feature))
    coverage = float(np.mean(covered))
    eligible = (
        len(event_bins) >= 25
        and coverage >= 0.8
        and peak["correlation"] >= 0.1
        and margin is not None
        and margin >= 0.02
    )
    result.update(
        {
            "peak": {**peak, "audio_lag_bins": int(coarse_local)},
            "alternative": alternative,
            "margin": margin,
            "coverage": coverage,
            "sub_bin_fraction": fraction,
            "offset_bins": coarse_local - start + fraction,
            "x_mean_bins": float(np.mean(event_bins[covered] + start)) if covered.any() else None,
            "weight": int(covered.sum()),
            "eligible": bool(eligible),
        }
    )
    return result


def fit_models(feature, fit_ms, cut, coarse):
    """No validation timestamps or letter labels are accepted by this API."""
    feature, fit_ms = np.asarray(feature, dtype=float), np.asarray(fit_ms, dtype=np.int64)
    if (
        cut < 8
        or not np.isfinite(feature).all()
        or np.any(fit_ms < 0)
        or np.any(fit_ms // 10 >= cut)
        or np.any(np.diff(fit_ms) < 0)
    ):
        raise ResearchError("Masukan fit waktu tidak valid.")
    bins = fit_ms // 10
    blocks = [
        block_offset(feature, bins, cut * q // 8, cut * (q + 1) // 8, coarse) for q in range(8)
    ]
    valid = [block for block in blocks if block["eligible"]]
    fallback = clock(coarse)
    models = {
        "legacy": fallback,
        "refined_constant": dict(fallback),
        "affine_policy": dict(fallback),
    }
    result = {
        "blocks": blocks,
        "eligible_blocks": len(valid),
        "regression": None,
        "refinement_admitted": False,
        "affine_admitted": False,
        "fallback_reason": "fewer than four eligible blocks",
        "models": models,
    }
    if len(valid) < 4:
        return result
    regression = weighted_clock_fit(
        [r["x_mean_bins"] for r in valid],
        [r["offset_bins"] for r in valid],
        [r["weight"] for r in valid],
    )
    result["regression"] = regression
    # cut is half the full number of log bins, including its possible odd last bin.
    if regression["x_span_bins"] < 0.5 * cut:
        result["fallback_reason"] = "eligible block span less than quarter of phase"
        return result
    constant = clock(regression["constant_a_bins"], source="weighted_block_constant")
    models["refined_constant"] = constant
    models["affine_policy"] = dict(constant)
    result["refinement_admitted"] = True
    if abs(regression["drift_ppm"]) > 200:
        result["fallback_reason"] = "affine drift outside 200 ppm"
    elif regression["max_abs_residual_ms"] > 15:
        result["fallback_reason"] = "affine block residual above 15 ms"
    else:
        models["affine_policy"] = clock(
            regression["a_bins"], regression["b"], "weighted_block_affine"
        )
        result.update({"affine_admitted": True, "fallback_reason": None})
    return result


def model_score(feature, pattern, event_bins, start, model, residual_bins=0):
    """Sample audio in log coordinates; never extrapolate beyond recorded features."""
    indices = np.arange(start, start + len(pattern), dtype=float)
    positions = model["a_bins"] + (1 + model["b"]) * (indices + residual_bins)
    finite = (positions >= 0) & (positions <= len(feature) - 1)
    mapped_events = model["a_bins"] + (1 + model["b"]) * (event_bins + start + residual_bins)
    covered = (mapped_events >= 0) & (mapped_events <= len(feature) - 1)
    count = int(covered.sum())
    score = None
    if finite.sum() >= 2:
        audio = np.interp(positions[finite], np.arange(len(feature)), feature)
        target = pattern[finite]
        audio, target = audio - audio.mean(), target - target.mean()
        denominator = float(np.sqrt(np.sum(audio * audio) * np.sum(target * target)))
        if denominator > 1e-10:
            score = float(np.clip(np.sum(audio * target) / denominator, -1, 1))
    return {
        "correlation": score,
        "covered_events": count,
        "coverage": count / len(event_bins) if len(event_bins) else 0.0,
        "overlap_bins": int(finite.sum()),
        "eligible": bool(
            score is not None
            and 2 * finite.sum() >= len(pattern)
            and len(event_bins)
            and 2 * count >= len(event_bins)
        ),
    }


def evaluate_models(feature, relative_ms, cut, total, selection, control_fits, val_controls):
    """Evaluate already selected clocks; validation residuals never update clocks."""
    bins = np.asarray(relative_ms, dtype=np.int64) // 10
    fit_bins, val_bins = bins[bins < cut], bins[bins >= cut] - cut
    fit_pattern, val_pattern = template(fit_bins, cut), template(val_bins, total - cut)
    results, arrays = {}, {}
    for method in METHODS:
        model = selection["models"][method]
        fit = model_score(feature, fit_pattern, fit_bins, 0, model)
        val = model_score(feature, val_pattern, val_bins, cut, model)
        local = [
            model_score(feature, val_pattern, val_bins, cut, model, residual_bins=d)
            for d in range(-25, 26)
        ]
        scores = np.array([v["correlation"] if v["eligible"] else np.nan for v in local])
        best = int(np.nanargmax(scores)) - 25 if np.isfinite(scores).any() else None
        null_fit = np.asarray([v[method] for v in control_fits], dtype=float)
        null_val = np.asarray(
            [model_score(feature, p, b, cut, model)["correlation"] for p, b in val_controls],
            dtype=float,
        )
        arrays[f"{method}_validation_local_curve"] = scores
        arrays[f"{method}_fit_null_scores"] = null_fit
        arrays[f"{method}_validation_null_scores"] = null_val
        fit_tail = (
            (1 + int(np.sum(null_fit >= fit["correlation"]))) / 20
            if fit["correlation"] is not None
            else None
        )
        val_tail = (
            (1 + int(np.sum(null_val >= val["correlation"]))) / 100
            if val["correlation"] is not None
            else None
        )
        flags = {
            "enough_events": len(fit_bins) >= 25 and len(val_bins) >= 25,
            "coverage": fit["coverage"] >= 0.8 and val["coverage"] >= 0.8,
            "correlation": fit["correlation"] is not None
            and fit["correlation"] >= 0.1
            and val["correlation"] is not None
            and val["correlation"] >= 0.1,
            "unique_peak": selection["global_peak_margin"] is not None
            and selection["global_peak_margin"] >= 0.02,
            "fit_controls": bool(np.isfinite(null_fit).all() and fit_tail == 0.05),
            "validation_controls": bool(np.isfinite(null_val).all() and val_tail == 0.01),
            "local_stability": best is not None and abs(best) <= 3,
        }
        results[method] = {
            "clock": model,
            "fit": fit,
            "validation": val,
            "validation_local_residual_ms": best * 10 if best is not None else None,
            "fit_control_tail_fraction": fit_tail,
            "validation_control_tail_fraction": val_tail,
            "criteria": flags,
            "provisional_stable_match": all(flags.values()),
        }
    return results, arrays


def physical_events(rows, stamps, phases, kinds, keys):
    """Reconstruct initial presses and release ownership across the complete log."""
    held, presses, releases = {}, [], []
    for row, stamp, phase, kind, key in zip(rows, stamps, phases, kinds, keys, strict=True):
        event = {"row": int(row), "timestamp_ms": int(stamp), "phase": str(phase), "key": str(key)}
        if kind == "press":
            if key not in held:
                held[key] = event
                presses.append(
                    {**event, "release_row": None, "release_ms": None, "release_phase": None}
                )
                held[key]["press_index"] = len(presses) - 1
        elif kind == "release":
            press = held.pop(key, None)
            releases.append({**event, "initial_press_row": press["row"] if press else None})
            if press:
                target = presses[press["press_index"]]
                target.update(
                    {"release_row": int(row), "release_ms": int(stamp), "release_phase": str(phase)}
                )
        else:
            raise ResearchError("Jenis event tidak dikenal.")
    return presses, releases


def window_audit(presses, releases, phase, origin, frames, rate, model):
    subset = [p for p in presses if p["phase"] == phase]
    rel = [r for r in releases if r["phase"] == phase]
    scale, offset = 1 + model["b"], model["a_bins"] / 100
    times = np.asarray([(p["timestamp_ms"] - origin) / 1000 for p in subset]) * scale + offset
    release_times = np.asarray([(r["timestamp_ms"] - origin) / 1000 for r in rel]) * scale + offset
    paired = np.asarray(
        [
            p["release_ms"] is not None
            and p["release_phase"] == phase
            and p["release_ms"] > p["timestamp_ms"]
            for p in subset
        ],
        dtype=bool,
    )
    own_release = np.asarray(
        [
            (p["release_ms"] - origin) / 1000 * scale + offset
            if p["release_ms"] is not None and p["release_phase"] == phase
            else np.nan
            for p in subset
        ]
    )
    press_frames = np.rint(times * rate).astype(np.int64)
    release_frames = np.rint(release_times * rate).astype(np.int64)
    own_release_frames = np.full(len(own_release), np.iinfo(np.int64).min, dtype=np.int64)
    known = np.isfinite(own_release)
    own_release_frames[known] = np.rint(own_release[known] * rate).astype(np.int64)
    arrays = {
        "press_row": np.asarray([p["row"] for p in subset], dtype=np.int64),
        "press_key": np.asarray([p["key"] for p in subset]),
        "press_audio_seconds": times,
        "press_audio_frame": press_frames,
        "same_phase_positive_release_pair": paired,
    }
    stats = []
    for length, pre in WINDOWS:
        start = press_frames - round(pre * rate / 1000)
        end = start + round(length * rate / 1000)
        complete = (start >= 0) & (end <= frames)
        press_count = (
            np.searchsorted(press_frames, end, side="left")
            - np.searchsorted(press_frames, start, side="left")
            - 1
        )
        release_count = np.searchsorted(release_frames, end, side="left") - np.searchsorted(
            release_frames, start, side="left"
        )
        own_inside = (own_release_frames >= start) & (own_release_frames < end)
        other_release = release_count - own_inside.astype(int)
        no_press = complete & (press_count == 0)
        no_other = no_press & (other_release == 0) & paired
        masks = {
            "complete": complete,
            "no_neighbor_press": no_press,
            "no_other_key_event": no_other,
        }
        for label, mask in masks.items():
            arrays[f"w{length}_{label}"] = mask
        arrays[f"w{length}_neighbor_press_count"] = press_count
        arrays[f"w{length}_other_release_count"] = other_release
        counts = {
            label: {
                str(k): int(np.sum(mask & (arrays["press_key"] == k)))
                for k in np.unique(arrays["press_key"])
            }
            for label, mask in masks.items()
        }
        stats.append(
            {
                "length_ms": length,
                "pre_ms": pre,
                "total_initial_press": len(times),
                "tiers": {
                    label: {"events": int(mask.sum()), "counts": counts[label]}
                    for label, mask in masks.items()
                },
            }
        )
    return stats, arrays
