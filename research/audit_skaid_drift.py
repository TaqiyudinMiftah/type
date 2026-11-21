"""Replay clock choices and fixed evaluations; verify regression and window counts directly."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, read_json, write_json
from ketik_audio.skaid_alignment import correlation_curve, permute_gaps, select_peak, template
from ketik_audio.skaid_drift import (
    METHODS,
    WINDOWS,
    evaluate_models,
    fit_models,
    model_score,
    physical_events,
    window_audit,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/skaid-drift"
NATIVE = ROOT / "reports/skaid-audit"
BASE = ROOT / "reports/skaid-alignment"


def verify(base, record):
    for name, digest in record["input_sha256"].items():
        if file_hash(ROOT / name) != digest:
            raise ResearchError(f"Masukan berubah: {name}")
    for name, digest in record["artifact_sha256"].items():
        if file_hash(base / name) != digest:
            raise ResearchError(f"Artefak berubah: {base / name}")


def independent_regression(selected):
    if selected["regression"] is None:
        return 0.0
    blocks = [b for b in selected["blocks"] if b["eligible"]]
    x, y, w = (
        np.array([b[k] for b in blocks], dtype=float)
        for k in ("x_mean_bins", "offset_bins", "weight")
    )
    center = float(np.mean(x))
    scale = float(np.ptp(x))
    design = np.column_stack((np.ones(len(x)), (x - center) / scale))
    coefficient = np.linalg.lstsq(design * np.sqrt(w[:, None]), y * np.sqrt(w), rcond=None)[0]
    slope = coefficient[1] / scale
    intercept = coefficient[0] - center * slope
    saved = selected["regression"]
    error = max(abs(intercept - saved["a_bins"]), abs(slope - saved["b"]))
    if error > 1e-8:
        raise ResearchError("Regresi tidak cocok dengan weighted least squares independen.")
    return float(error)


def verify_release_ownership(presses, events, prefix):
    release_indices = np.flatnonzero(events[f"{prefix}_event"] == "release")
    keys, rows = events[f"{prefix}_physical_key"], events[f"{prefix}_row"]
    stamps, phases = events[f"{prefix}_timestamp_ms"], events[f"{prefix}_phase"]
    by_key = {k: release_indices[keys[release_indices] == k] for k in np.unique(keys)}
    for press in presses:
        candidates = by_key[press["key"]]
        later = int(np.searchsorted(rows[candidates], press["row"], side="right"))
        if later == len(candidates):
            expected = (None, None, None)
        else:
            i = candidates[later]
            expected = (
                int(rows[i]),
                int(stamps[i]),
                str(phases[i]),
            )
        actual = (press["release_row"], press["release_ms"], press["release_phase"])
        if actual != expected:
            raise ResearchError("Kepemilikan release berbeda dari next-release search.")


def direct_windows(presses, releases, phase, origin, audio, model, saved):
    p = [v for v in presses if v["phase"] == phase]
    r = [v for v in releases if v["phase"] == phase]
    rate = audio["sample_rate"]

    def frame(stamp):
        return int(
            np.rint(((stamp - origin) / 1000 * (1 + model["b"]) + model["a_bins"] / 100) * rate)
        )

    press_frames = np.array([frame(v["timestamp_ms"]) for v in p])
    release_frames = np.array([frame(v["timestamp_ms"]) for v in r])
    owners = np.array([v["initial_press_row"] or -1 for v in r])
    press_indices = np.arange(len(p))
    sample = np.unique(np.rint(np.linspace(0, len(p) - 1, min(50, len(p)))).astype(int))
    count = 0
    for i in sample:
        for length, pre in WINDOWS:
            start = press_frames[i] - round(pre * rate / 1000)
            end = start + round(length * rate / 1000)
            complete = start >= 0 and end <= audio["frames"]
            other_press = int(
                np.count_nonzero(
                    (press_frames >= start) & (press_frames < end) & (press_indices != i)
                )
            )
            other_release = int(
                np.count_nonzero(
                    (release_frames >= start) & (release_frames < end) & (owners != p[i]["row"])
                )
            )
            paired = (
                p[i]["release_ms"] is not None
                and p[i]["release_phase"] == phase
                and p[i]["release_ms"] > p[i]["timestamp_ms"]
            )
            expected = {
                "complete": complete,
                "no_neighbor_press": complete and other_press == 0,
                "no_other_key_event": complete
                and other_press == 0
                and other_release == 0
                and paired,
            }
            for key, value in expected.items():
                if bool(saved[f"w{length}_{key}"][i]) != bool(value):
                    raise ResearchError("Mask jendela berbeda dari scan interval langsung.")
            if (
                int(saved[f"w{length}_neighbor_press_count"][i]) != other_press
                or int(saved[f"w{length}_other_release_count"][i]) != other_release
            ):
                raise ResearchError("Hitungan event tetangga berbeda.")
            count += 1
    return count


def run(output):
    results = read_json(RUN / "results.json")
    for base, record in (
        (NATIVE, read_json(NATIVE / "summary.json")),
        (BASE, read_json(BASE / "summary.json")),
        (RUN, results),
    ):
        verify(base, record)
    selection = read_json(RUN / "selection.json")
    if file_hash(RUN / "selection.json") != results["selection_sha256"]:
        raise ResearchError("Seleksi berubah.")
    if selection["created_at"] >= results["evaluation_started_at"]:
        raise ResearchError("Seleksi tidak dikunci sebelum evaluasi.")
    fresh_directory(output)
    paths = [Path(__file__), RUN / "results.json", RUN / "selection.json"]
    signature = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    write_json(output / "lock.json", {"created_at": now(), "input_sha256": signature})
    selected = {r["pair_index"]: r for r in selection["records"]}
    logs = {r["session_tokens"][0]: r for r in read_json(NATIVE / "logs.json")["logs"]}
    inventory = {r["path"]: r["audio"] for r in read_json(NATIVE / "inventory.json")["files"]}
    checks, max_regression_error, direct_count = [], 0.0, 0
    with np.load(NATIVE / "events.npz", allow_pickle=False) as events:
        physical = {}
        for token, log in logs.items():
            prefix = log["event_array_prefix"]
            physical[token] = physical_events(
                *(
                    events[f"{prefix}_{key}"]
                    for key in ("row", "timestamp_ms", "phase", "event", "physical_key")
                )
            )
            verify_release_ownership(physical[token][0], events, prefix)
        for record in read_json(RUN / "pairs.json")["records"]:
            if record["status"] != "assessed":
                continue
            index = record["pair_index"]
            chosen = selected[index]
            with (
                np.load(BASE / "arrays" / f"pair{index:02d}.npz", allow_pickle=False) as baseline,
                np.load(RUN / "arrays" / f"pair{index:02d}.npz", allow_pickle=False) as saved,
            ):
                feature, relative = baseline["onset"], baseline["relative_press_ms"]
                cut, total = chosen["fit_bins"], chosen["total_bins"]
                fit_ms, val_ms = relative[relative // 10 < cut], relative[relative // 10 >= cut]
                actual = fit_models(
                    feature, fit_ms, cut, int(chosen["selection"]["models"]["legacy"]["a_bins"])
                )
                for key, value in actual.items():
                    if value != chosen["selection"][key]:
                        raise ResearchError(f"Seleksi pair{index:02d} berbeda: {key}")
                max_regression_error = max(max_regression_error, independent_regression(actual))
                rng = np.random.default_rng(chosen["seed"])
                control_scores = []
                for control in chosen["controls"]:
                    ms = permute_gaps(fit_ms, rng)
                    bins = ms // 10
                    pattern = template(bins, cut)
                    peak = select_peak(correlation_curve(feature, pattern, bins))
                    replay = fit_models(feature, ms, cut, peak["lag_bins"])
                    scores = {
                        m: model_score(feature, pattern, bins, 0, replay["models"][m])[
                            "correlation"
                        ]
                        for m in METHODS
                    }
                    if (
                        replay != control["selection"]
                        or scores != control["fit_scores"]
                        or peak != control["coarse_peak"]
                    ):
                        raise ResearchError("Seleksi/skor kontrol fit berbeda.")
                    max_regression_error = max(max_regression_error, independent_regression(replay))
                    control_scores.append(scores)
                if rng.bit_generator.state != chosen["validation_rng_state"]:
                    raise ResearchError("State RNG validasi berbeda.")
                controls = []
                for _ in range(99):
                    bins = permute_gaps(val_ms, rng) // 10 - cut
                    controls.append((template(bins, total - cut), bins))
                methods, arrays = evaluate_models(
                    feature, relative, cut, total, chosen["selection"], control_scores, controls
                )
                if methods != record["methods"]:
                    raise ResearchError("Metrik akhir berbeda pada replay.")
                for key, array in arrays.items():
                    if not np.array_equal(array, saved[key], equal_nan=True):
                        raise ResearchError("Array evaluasi berbeda.")
                presses, releases = physical[record["session_token"]]
                audio = inventory[chosen["audio"]]
                for method in METHODS:
                    stats, crops = window_audit(
                        presses,
                        releases,
                        record["phase"],
                        chosen["log_origin_ms"],
                        audio["frames"],
                        audio["sample_rate"],
                        methods[method]["clock"],
                    )
                    if stats != record["windows"][method]:
                        raise ResearchError("Agregat jendela berbeda pada replay.")
                    for key, array in crops.items():
                        if not np.array_equal(array, saved[f"{method}_{key}"]):
                            raise ResearchError("Mask jendela berbeda pada replay.")
                    direct_count += direct_windows(
                        presses,
                        releases,
                        record["phase"],
                        chosen["log_origin_ms"],
                        audio,
                        methods[method]["clock"],
                        crops,
                    )
                checks.append(
                    {
                        "pair_index": index,
                        "selection_exact": True,
                        "all_19_control_choices_exact": True,
                        "fixed_evaluation_exact": True,
                        "window_arrays_and_counts_exact": True,
                    }
                )
            print(f"Audited drift pair{index:02d}", flush=True)
    verify(RUN, results)
    for name, digest in signature.items():
        if file_hash(ROOT / name) != digest:
            raise ResearchError("Kode/masukan audit berubah.")
    write_json(output / "checks.json", {"pairs": checks})
    result = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": signature,
        "pairs_replayed": len(checks),
        "clock_pipelines_replayed": len(checks) * 20,
        "method_evaluations_replayed": len(checks) * 3,
        "direct_interval_checks": direct_count,
        "all_results_exact": True,
        "max_independent_regression_error_bins": max_regression_error,
        "scope": "numerical and event-log interval consistency; not manual onset ground truth",
    }
    result["artifact_sha256"] = {
        str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()
    }
    write_json(output / "audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/skaid-drift-audit")
    args = parser.parse_args()
    try:
        run(args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
