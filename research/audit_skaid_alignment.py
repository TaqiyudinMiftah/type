"""Replay timing diagnostics and independently check events, templates and correlations."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from ketik_audio.common import (
    ResearchError,
    file_hash,
    fresh_directory,
    now,
    read_json,
    write_json,
)
from ketik_audio.skaid_alignment import assess_timing

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "reports/skaid-audit"
ALIGNMENT = ROOT / "reports/skaid-alignment"


def verify_manifest(base, summary):
    for name, digest in summary["input_sha256"].items():
        if file_hash(ROOT / name) != digest:
            raise ResearchError(f"Masukan berubah: {name}")
    for name, digest in summary["artifact_sha256"].items():
        if file_hash(base / name) != digest:
            raise ResearchError(f"Artefak berubah: {base / name}")


def initial_press_rows(events, prefix):
    active, rows = set(), []
    raw = events[f"{prefix}_raw_key"]
    keys = [str(k).lower() if len(k) == 1 and k.isascii() and k.isalpha() else str(k) for k in raw]
    if keys != events[f"{prefix}_physical_key"].tolist():
        raise ResearchError("Canonical physical key tidak cocok.")
    for row, kind, key in zip(
        events[f"{prefix}_row"], events[f"{prefix}_event"], keys, strict=True
    ):
        if kind == "press":
            if key not in active:
                rows.append(row)
            active.add(key)
        else:
            active.discard(key)
    return np.asarray(rows, dtype=np.int64)


def direct_correlation(audio, pattern, lag):
    lo, hi = max(0, -lag), min(len(pattern), len(audio) - lag)
    if hi - lo < 2:
        return None
    a, p = audio[lo + lag : hi + lag], pattern[lo:hi]
    a, p = a - a.mean(), p - p.mean()
    denominator = np.sqrt(np.sum(a * a) * np.sum(p * p))
    return float(np.sum(a * p) / denominator) if denominator > 1e-10 else None


def independent_pattern(bins, length):
    impulse = np.zeros(length)
    np.add.at(impulse, bins, 1)
    kernel = np.exp(-0.5 * np.arange(-3, 4, dtype=float) ** 2)
    return np.convolve(impulse, kernel / kernel.sum(), mode="same")


def run(output):
    native, alignment = read_json(NATIVE / "summary.json"), read_json(ALIGNMENT / "summary.json")
    verify_manifest(NATIVE, native)
    verify_manifest(ALIGNMENT, alignment)
    fresh_directory(output)
    signature = {
        "reports/skaid-audit/summary.json": file_hash(NATIVE / "summary.json"),
        "reports/skaid-alignment/summary.json": file_hash(ALIGNMENT / "summary.json"),
        "research/audit_skaid_alignment.py": file_hash(Path(__file__)),
    }
    write_json(output / "lock.json", {"created_at": now(), "input_sha256": signature})
    results = read_json(ALIGNMENT / "pairs.json")["pairs"]
    logs = read_json(NATIVE / "logs.json")["logs"]
    inventory = {r["path"]: r for r in read_json(NATIVE / "inventory.json")["files"]}
    logs_by_id = {r["session_tokens"][0]: r for r in logs}
    checks, max_array_error, max_direct_error, max_template_error = [], 0.0, 0.0, 0.0
    with (
        np.load(NATIVE / "events.npz", allow_pickle=False) as events,
        np.load(NATIVE / "envelopes.npz", allow_pickle=False) as env,
    ):
        initial_by_id = {
            k: initial_press_rows(events, v["event_array_prefix"]) for k, v in logs_by_id.items()
        }
        for result in results:
            if result["status"] != "assessed":
                continue
            index, token, phase = result["pair_index"], result["session_token"], result["phase"]
            log = logs_by_id[token]
            prefix = log["event_array_prefix"]
            select = (events[f"{prefix}_phase"] == phase) & np.isin(
                events[f"{prefix}_row"], initial_by_id[token]
            )
            rows = events[f"{prefix}_row"][select]
            stamps = events[f"{prefix}_timestamp_ms"][select]
            origin = log["phases"][phase]["min_timestamp_ms"]
            relative = stamps - origin
            audio = inventory[result["audio"]]
            rms = env[f"{audio['envelope_prefix']}_rms"]
            replay, arrays = assess_timing(rms, relative, result["span_ms"], result["seed"])
            for key, value in replay.items():
                if result[key] != value:
                    raise ResearchError(f"Replay metrik berbeda: pair{index:02d}, {key}")
            with np.load(
                ALIGNMENT / "arrays" / f"pair{index:02d}.npz", allow_pickle=False
            ) as saved:
                if not np.array_equal(rows, saved["press_row"]):
                    raise ResearchError("Initial press row berbeda pada replay independen.")
                for key, values in arrays.items():
                    if values.shape != saved[key].shape or not np.array_equal(
                        values, saved[key], equal_nan=True
                    ):
                        raise ResearchError(f"Array berbeda: pair{index:02d}, {key}")
                    if np.issubdtype(values.dtype, np.number) and len(values):
                        finite = np.isfinite(values)
                        if finite.any():
                            max_array_error = max(
                                max_array_error,
                                float(np.max(np.abs(values[finite] - saved[key][finite]))),
                            )
                # Independent convolution rather than gaussian_filter1d.
                cut, total = result["fit_bins"], result["total_bins"]
                bins = relative // 10
                for name, start, end in (
                    ("fit_template", 0, cut),
                    ("validation_template", cut, total),
                ):
                    selected = bins[(bins >= start) & (bins < end)] - start
                    independent = independent_pattern(selected, end - start)
                    error = float(np.max(np.abs(independent - saved[name])))
                    max_template_error = max(max_template_error, error)
                    if error > 1e-12:
                        raise ResearchError("Konvolusi template tidak cocok.")
                # Direct centered dot product, no FFT or cumulative-sum normalization.
                points = []
                for label, peak in (
                    ("selected", result["fit_peak"]),
                    ("alternative", result["alternative_peak"]),
                ):
                    if peak:
                        direct = direct_correlation(
                            saved["onset"], saved["fit_template"], peak["lag_bins"]
                        )
                        delta = abs(direct - peak["correlation"]) if direct is not None else np.inf
                        max_direct_error = max(max_direct_error, delta)
                        if delta > 1e-10:
                            raise ResearchError("Korelasi langsung berbeda dari FFT.")
                        points.append(
                            {
                                "point": label,
                                "direct": direct,
                                "stored": peak["correlation"],
                                "error": delta,
                            }
                        )
                val = direct_correlation(
                    saved["onset"],
                    saved["validation_template"],
                    result["fit_peak"]["lag_bins"] + cut,
                )
                stored_val = result["validation"]["correlation"]
                if (val is None) != (stored_val is None) or (
                    val is not None and abs(val - stored_val) > 1e-10
                ):
                    raise ResearchError("Korelasi validasi langsung berbeda.")
                if val is not None:
                    max_direct_error = max(max_direct_error, abs(val - stored_val))
                # Class counts are descriptive only; count again from audited rows.
                keys = events[f"{prefix}_physical_key"][select][saved["press_in_audio"]]
                counts = dict(
                    sorted(Counter(str(k) for k in keys if len(k) == 1 and "a" <= k <= "z").items())
                )
                if counts != result["provisional_in_audio_letter_counts"]:
                    raise ResearchError("Hitungan huruf deskriptif berbeda.")
                checks.append(
                    {
                        "pair_index": index,
                        "initial_rows": len(rows),
                        "replay_exact": True,
                        "independent_template": True,
                        "direct_fit": points,
                        "direct_validation": val,
                    }
                )
            print(
                f"Replayed timing pair{index:02d} with independent row/template/dot checks",
                flush=True,
            )
    verify_manifest(NATIVE, native)
    verify_manifest(ALIGNMENT, alignment)
    for name, digest in signature.items():
        if file_hash(ROOT / name) != digest:
            raise ResearchError("Kode/masukan berubah selama replay.")
    write_json(output / "checks.json", {"checks": checks})
    audit = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": signature,
        "pairs_replayed": len(checks),
        "all_metrics_exact": True,
        "all_arrays_exact": True,
        "max_array_error": max_array_error,
        "max_independent_template_error": max_template_error,
        "max_direct_correlation_error": max_direct_error,
        "scope": (
            "timing reproducibility and numerical/state checks; does not certify physical "
            "event onsets or independence of microphone/session"
        ),
    }
    audit["artifact_sha256"] = {
        str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()
    }
    write_json(output / "audit.json", audit)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/skaid-alignment-audit")
    args = parser.parse_args()
    try:
        run(args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
