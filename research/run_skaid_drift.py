"""Select clocks on early timing, lock all choices, then evaluate later timing and windows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ketik_audio.common import (
    ResearchError,
    environment,
    file_hash,
    fresh_directory,
    now,
    read_json,
    write_json,
)
from ketik_audio.skaid_alignment import correlation_curve, permute_gaps, select_peak, template
from ketik_audio.skaid_drift import (
    METHODS,
    evaluate_models,
    fit_models,
    model_score,
    physical_events,
    window_audit,
)

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "reports/skaid-audit"
BASE = ROOT / "reports/skaid-alignment"
REPLAY = ROOT / "reports/skaid-alignment-audit"
PROTOCOL = ROOT / "research/SKAID_DRIFT_PROTOCOL.md"


def verify(base, record):
    for name, digest in record["input_sha256"].items():
        if file_hash(ROOT / name) != digest:
            raise ResearchError(f"Masukan berubah: {name}")
    for name, digest in record["artifact_sha256"].items():
        if file_hash(base / name) != digest:
            raise ResearchError(f"Artefak berubah: {base / name}")


def signature():
    for base, name in ((NATIVE, "summary.json"), (BASE, "summary.json"), (REPLAY, "audit.json")):
        verify(base, read_json(base / name))
    paths = [
        PROTOCOL,
        Path(__file__),
        ROOT / "src/ketik_audio/skaid_drift.py",
        ROOT / "src/ketik_audio/skaid_alignment.py",
        ROOT / "uv.lock",
        NATIVE / "summary.json",
        BASE / "summary.json",
        BASE / "pairs.json",
        REPLAY / "audit.json",
    ]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def select(output):
    sig = signature()
    fresh_directory(output)
    write_json(
        output / "selection-lock.json",
        {"created_at": now(), "input_sha256": sig, "environment": environment(), "device": "cpu"},
    )
    (output / "protocol.md").write_bytes(PROTOCOL.read_bytes())
    records = []
    for old in read_json(BASE / "pairs.json")["pairs"]:
        index = old["pair_index"]
        record = {
            "pair_index": index,
            "session_token": old["session_token"],
            "phase": old["phase"],
            "status": old["status"],
        }
        if old["status"] != "assessed":
            record["reason"] = old["reason"]
            records.append(record)
            continue
        with np.load(BASE / "arrays" / f"pair{index:02d}.npz", allow_pickle=False) as arrays:
            feature, relative = arrays["onset"], arrays["relative_press_ms"]
            cut = old["fit_bins"]
            fit_ms = relative[relative // 10 < cut]
            selected = fit_models(feature, fit_ms, cut, old["fit_peak"]["lag_bins"])
            selected["global_peak_margin"] = old["fit_peak_margin"]
            rng = np.random.default_rng(old["seed"])
            controls = []
            for _ in range(19):
                ms = permute_gaps(fit_ms, rng)
                bins = ms // 10
                pattern = template(bins, cut)
                peak = select_peak(correlation_curve(feature, pattern, bins))
                if peak is None:
                    raise ResearchError("Kontrol tidak memiliki puncak kasar.")
                control = fit_models(feature, ms, cut, peak["lag_bins"])
                scores = {
                    m: model_score(feature, pattern, bins, 0, control["models"][m])["correlation"]
                    for m in METHODS
                }
                controls.append({"selection": control, "fit_scores": scores, "coarse_peak": peak})
            record.update(
                {
                    "audio": old["audio"],
                    "log_origin_ms": old["log_origin_ms"],
                    "fit_bins": cut,
                    "total_bins": old["total_bins"],
                    "seed": old["seed"],
                    "fit_event_count": len(fit_ms),
                    "selection": selected,
                    "controls": controls,
                    "validation_rng_state": rng.bit_generator.state,
                }
            )
        records.append(record)
        print(
            f"Selected pair{index:02d}: {record['selection']['eligible_blocks']} blocks, "
            f"{record['selection']['models']['affine_policy']['source']}",
            flush=True,
        )
    if signature() != sig:
        raise ResearchError("Masukan/kode berubah selama seleksi.")
    result = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": sig,
        "records": records,
        "all_choices_locked_before_evaluation": True,
    }
    write_json(output / "selection.json", result)
    return result


def check_selection(output):
    selection = read_json(output / "selection.json")
    if selection["status"] != "complete" or not selection["all_choices_locked_before_evaluation"]:
        raise ResearchError("Seleksi belum lengkap.")
    if signature() != selection["input_sha256"]:
        raise ResearchError("Masukan/kode berbeda dari seleksi terkunci.")
    return selection


def feasibility(records):
    rows = []
    for method in METHODS:
        good = [
            r
            for r in records
            if r["status"] == "assessed" and r["methods"][method]["provisional_stable_match"]
        ]
        by_id = {
            r["session_token"]: {
                p["phase"]: p for p in good if p["session_token"] == r["session_token"]
            }
            for r in good
        }
        by_id = {t: d for t, d in by_id.items() if len(d) == 2}
        for wi, length in enumerate((400, 200, 120, 80)):
            for tier in ("complete", "no_neighbor_press", "no_other_key_event"):
                for vocab, letters in (
                    ("letters26", "abcdefghijklmnopqrstuvwxyz"),
                    ("exploratory12", "etaoinshrdlu"),
                ):
                    for query_min in (1, 5):
                        for k in (1, 2, 3, 5, 10):
                            eligible = []
                            for token, phases in by_id.items():
                                counts = {
                                    p: r["windows"][method][wi]["tiers"][tier]["counts"]
                                    for p, r in phases.items()
                                }
                                if all(
                                    counts["Phase 1"].get(c, 0) >= k
                                    and counts["Phase 2"].get(c, 0) >= query_min
                                    for c in letters
                                ):
                                    eligible.append(token)
                            rows.append(
                                {
                                    "method": method,
                                    "length_ms": length,
                                    "tier": tier,
                                    "vocabulary": vocab,
                                    "query_min_per_class": query_min,
                                    "K": k,
                                    "both_stable_ids": len(by_id),
                                    "eligible_ids": len(eligible),
                                    "session_tokens": eligible,
                                }
                            )
    return rows


def evaluate(output):
    selection = check_selection(output)
    if (output / "evaluation-lock.json").exists():
        raise ResearchError("Evaluasi sudah dimulai; gunakan direktori run baru.")
    selection_hash = file_hash(output / "selection.json")
    started = now()
    write_json(
        output / "evaluation-lock.json",
        {
            "created_at": started,
            "selection_sha256": selection_hash,
            "input_sha256": selection["input_sha256"],
        },
    )
    (output / "arrays").mkdir()
    old_by_index = {r["pair_index"]: r for r in read_json(BASE / "pairs.json")["pairs"]}
    logs = {r["session_tokens"][0]: r for r in read_json(NATIVE / "logs.json")["logs"]}
    inventory = {r["path"]: r for r in read_json(NATIVE / "inventory.json")["files"]}
    records, maximum_legacy_error = [], 0.0
    with np.load(NATIVE / "events.npz", allow_pickle=False) as events:
        physical = {}
        for token, log in logs.items():
            p = log["event_array_prefix"]
            physical[token] = physical_events(
                *(
                    events[f"{p}_{key}"]
                    for key in ("row", "timestamp_ms", "phase", "event", "physical_key")
                )
            )
        for selected in selection["records"]:
            index = selected["pair_index"]
            record = {k: selected[k] for k in ("pair_index", "session_token", "phase", "status")}
            if selected["status"] != "assessed":
                record["reason"] = selected["reason"]
                records.append(record)
                continue
            old = old_by_index[index]
            with np.load(BASE / "arrays" / f"pair{index:02d}.npz", allow_pickle=False) as data:
                feature, relative = data["onset"], data["relative_press_ms"]
                cut, total = selected["fit_bins"], selected["total_bins"]
                val_ms = relative[relative // 10 >= cut]
                rng = np.random.default_rng()
                rng.bit_generator.state = selected["validation_rng_state"]
                controls = []
                for _ in range(99):
                    bins = permute_gaps(val_ms, rng) // 10 - cut
                    controls.append((template(bins, total - cut), bins))
                methods, arrays = evaluate_models(
                    feature,
                    relative,
                    cut,
                    total,
                    selected["selection"],
                    [r["fit_scores"] for r in selected["controls"]],
                    controls,
                )
                legacy = methods["legacy"]
                for actual, expected in (
                    (legacy["fit"]["correlation"], old["fit_peak"]["correlation"]),
                    (legacy["validation"]["correlation"], old["validation"]["correlation"]),
                ):
                    if (actual is None) != (expected is None):
                        raise ResearchError("Baseline legacy berbeda dalam finite status.")
                    error = abs(actual - expected) if actual is not None else 0.0
                    maximum_legacy_error = max(maximum_legacy_error, error)
                    if error > 1e-10:
                        raise ResearchError("Skor baseline legacy berbeda.")
                for key, saved in (
                    ("legacy_fit_null_scores", "fit_null_max_correlation"),
                    ("legacy_validation_null_scores", "validation_null_fixed_correlation"),
                ):
                    if not np.allclose(
                        arrays[key], data[saved], rtol=0, atol=1e-10, equal_nan=True
                    ):
                        raise ResearchError("Kontrol baseline legacy berbeda.")
                if legacy["criteria"] != old["criteria"]:
                    raise ResearchError("Flag baseline legacy berbeda.")
                presses, releases = physical[selected["session_token"]]
                phase_presses = [p for p in presses if p["phase"] == selected["phase"]]
                if [p["row"] for p in phase_presses] != data["press_row"].tolist():
                    raise ResearchError("Row initial press tidak cocok.")
                audio = inventory[selected["audio"]]["audio"]
                windows = {}
                for method in METHODS:
                    stats, crops = window_audit(
                        presses,
                        releases,
                        selected["phase"],
                        selected["log_origin_ms"],
                        audio["frames"],
                        audio["sample_rate"],
                        methods[method]["clock"],
                    )
                    windows[method] = stats
                    arrays.update({f"{method}_{k}": v for k, v in crops.items()})
                record.update({"methods": methods, "windows": windows})
                np.savez_compressed(output / "arrays" / f"pair{index:02d}.npz", **arrays)
            records.append(record)
            print(
                f"Evaluated pair{index:02d}: "
                + ", ".join(f"{m}={methods[m]['provisional_stable_match']}" for m in METHODS),
                flush=True,
            )
    eligibility = feasibility(records)
    write_json(output / "pairs.json", {"records": records})
    write_json(output / "label-feasibility.json", {"rows": eligibility})
    with (output / "label-feasibility.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(eligibility[0]))
        writer.writeheader()
        writer.writerows(
            {**r, "session_tokens": ";".join(r["session_tokens"])} for r in eligibility
        )
    csv_rows = []
    for r in records:
        if r["status"] != "assessed":
            continue
        for method in METHODS:
            m = r["methods"][method]
            csv_rows.append(
                {
                    "pair_index": r["pair_index"],
                    "session_token": r["session_token"],
                    "phase": r["phase"],
                    "method": method,
                    **m["clock"],
                    "fit_correlation": m["fit"]["correlation"],
                    "validation_correlation": m["validation"]["correlation"],
                    "fit_coverage": m["fit"]["coverage"],
                    "validation_coverage": m["validation"]["coverage"],
                    "validation_local_residual_ms": m["validation_local_residual_ms"],
                    "provisional_stable_match": m["provisional_stable_match"],
                }
            )
    with (output / "timing.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    if (
        file_hash(output / "selection.json") != selection_hash
        or check_selection(output) != selection
    ):
        raise ResearchError("Seleksi berubah selama evaluasi.")
    result = {
        "created_at": now(),
        "status": "complete",
        "evaluation_started_at": started,
        "selection_locked_at": selection["created_at"],
        "selection_sha256": selection_hash,
        "input_sha256": selection["input_sha256"],
        "pairs": len(records),
        "assessed_pairs": sum(r["status"] == "assessed" for r in records),
        "stable_pairs": {
            m: sum(
                r["status"] == "assessed" and r["methods"][m]["provisional_stable_match"]
                for r in records
            )
            for m in METHODS
        },
        "maximum_legacy_correlation_error": maximum_legacy_error,
        "legacy_controls_and_flags_match": True,
        "models_trained_or_evaluated": 0,
        "segmentation_certified": False,
    }
    result["artifact_sha256"] = {
        str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()
    }
    write_json(output / "results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "evaluate"))
    parser.add_argument("--output", type=Path, default=ROOT / "runs/skaid-drift")
    args = parser.parse_args()
    try:
        (select if args.stage == "select" else evaluate)(args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
