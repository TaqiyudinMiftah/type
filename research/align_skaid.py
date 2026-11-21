"""Audit timing patterns with a fixed offset and later-log validation; CPU only."""

from __future__ import annotations

import argparse
import csv
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from ketik_audio.common import (
    ResearchError,
    environment,
    file_hash,
    fresh_directory,
    now,
    read_json,
    write_json,
)
from ketik_audio.skaid import decode, pcm_fingerprint, probe
from ketik_audio.skaid_alignment import assess_timing, waveform_copy_metrics

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "reports/skaid-audit"
PROTOCOL = ROOT / "research/SKAID_ALIGNMENT_PROTOCOL.md"
SOURCE = ROOT / "data/external/skaid-v1/recordings.zip"


def signature():
    audit = read_json(AUDIT / "summary.json")
    if audit["status"] != "complete" or audit["decode_failures"]:
        raise ResearchError("Audit native belum lengkap/ada kegagalan dekode.")
    for name, digest in audit["input_sha256"].items():
        if file_hash(ROOT / name) != digest:
            raise ResearchError(f"Sumber audit native berubah: {name}")
    for name, digest in audit["artifact_sha256"].items():
        if file_hash(AUDIT / name) != digest:
            raise ResearchError(f"Artefak audit native berubah: {name}")
    paths = [
        PROTOCOL,
        Path(__file__),
        ROOT / "src/ketik_audio/skaid_alignment.py",
        AUDIT / "summary.json",
        AUDIT / "inventory.json",
        AUDIT / "logs.json",
        AUDIT / "pairing.json",
        AUDIT / "events.npz",
        AUDIT / "envelopes.npz",
        ROOT / "uv.lock",
    ]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def duplicate_check(inventory):
    wavs = [r for r in inventory if r["identity"]["extension"] == "wav"]
    if len(wavs) != 1:
        raise ResearchError("Jumlah WAV tambahan berubah.")
    wav = wavs[0]
    identity = wav["identity"]
    candidates = [
        r
        for r in inventory
        if r["identity"]["session_token"] == identity["session_token"]
        and r["identity"]["phase"] == identity["phase"]
        and r["identity"]["extension"] == "m4a"
    ]
    if len(candidates) != 1:
        raise ResearchError("M4A pasangan WAV tidak unik.")
    m4a = candidates[0]
    decoded = []
    with (
        zipfile.ZipFile(SOURCE) as archive,
        tempfile.TemporaryDirectory(prefix="skaid-copy-") as temporary,
    ):
        for i, row in enumerate((m4a, wav)):
            path = Path(temporary) / f"audio{i}.{row['identity']['extension']}"
            path.write_bytes(archive.read(row["path"]))
            _, stream = probe(path)
            values, rate = decode(path, stream)
            if pcm_fingerprint(values, rate) != row["audio"]["pcm_sha256"]:
                raise ResearchError("Hasil decode ulang berbeda dari audit native.")
            decoded.append((values, rate))
    (reference, source_rate), (candidate, target_rate) = decoded
    if (
        source_rate != 48000
        or target_rate != 16000
        or reference.shape[1] != 1
        or candidate.shape[1] != 1
    ):
        raise ResearchError("Laju/kanal pasangan salinan berbeda dari protokol.")
    resampled = resample_poly(
        reference[:, 0].astype(np.float64), 1, 3, window=("kaiser", 5.0), padtype="constant"
    )
    return {
        "m4a": m4a["path"],
        "wav": wav["path"],
        "source_rate": source_rate,
        "target_rate": target_rate,
        "resampling": {"up": 1, "down": 3, "window": ["kaiser", 5.0], "padtype": "constant"},
        "m4a_seconds": len(reference) / source_rate,
        "wav_seconds": len(candidate) / target_rate,
        "resampled_frames": len(resampled),
        "wav_frames": len(candidate),
        **waveform_copy_metrics(resampled, candidate[:, 0], target_rate),
    }


def run(output):
    sig = signature()
    fresh_directory(output)
    (output / "arrays").mkdir()
    write_json(
        output / "lock.json",
        {"created_at": now(), "input_sha256": sig, "device": "cpu", "environment": environment()},
    )
    (output / "protocol.md").write_bytes(PROTOCOL.read_bytes())
    inventory = read_json(AUDIT / "inventory.json")["files"]
    by_audio = {r["path"]: r for r in inventory}
    logs = {r["file"]: r for r in read_json(AUDIT / "logs.json")["logs"]}
    copy = duplicate_check(inventory)
    write_json(output / "transformed-copy.json", copy)
    results, csv_rows = [], []
    with (
        np.load(AUDIT / "events.npz", allow_pickle=False) as events,
        np.load(AUDIT / "envelopes.npz", allow_pickle=False) as envelopes,
    ):
        for index, pair in enumerate(read_json(AUDIT / "pairing.json")["pairs"]):
            result = {
                "pair_index": index,
                "session_token": pair["session_token"],
                "phase": pair["phase"],
                "status": "not_assessed",
                "provisional_stable_match": False,
            }
            if len(pair["canonical_m4a_candidates"]) != 1:
                result["reason"] = "missing or nonunique canonical M4A"
            elif not pair["timestamp_clock_usable"]:
                result["reason"] = "event timestamps collapsed/nonmonotonic"
            else:
                name = pair["canonical_m4a_candidates"][0]
                row, log = by_audio[name], logs[pair["log"]]
                prefix = log["event_array_prefix"]
                selected = (events[f"{prefix}_phase"] == pair["phase"]) & (
                    events[f"{prefix}_event"] == "press"
                )
                repeated = np.asarray(
                    [r["row"] for r in log["repeated_press_while_held"]], dtype=np.int64
                )
                selected &= ~np.isin(events[f"{prefix}_row"], repeated)
                origin = pair["log_phase"]["min_timestamp_ms"]
                stamps = events[f"{prefix}_timestamp_ms"][selected]
                rms = envelopes[f"{row['envelope_prefix']}_rms"]
                timing, arrays = assess_timing(
                    rms, stamps - origin, pair["log_phase"]["span_ms"], 17282184 + index
                )
                arrays["press_row"] = events[f"{prefix}_row"][selected]
                anchors = [
                    r["first_log_minus_tag_seconds"]
                    for r in pair["metadata_anchor_candidates"]
                    if r["path"] == name and r["scope"] == "format" and r["tag"] == "creation_time"
                ]
                result.update(
                    {
                        "status": "assessed",
                        "audio": name,
                        "log_origin_ms": origin,
                        "metadata_first_log_minus_creation_seconds": anchors[0]
                        if len(anchors) == 1
                        else None,
                        "audio_duration_seconds": row["audio"]["duration_seconds"],
                        **timing,
                    }
                )
                if "press_in_audio" in arrays:
                    in_audio = arrays["press_in_audio"]
                    keys = events[f"{prefix}_physical_key"][selected][in_audio]
                    result["provisional_in_audio_letter_counts"] = dict(
                        sorted(
                            Counter(str(k) for k in keys if len(k) == 1 and "a" <= k <= "z").items()
                        )
                    )
                    all_gaps = np.diff(stamps)
                    result["consecutive_press_gaps"] = {
                        "count": len(all_gaps),
                        "median_ms": float(np.median(all_gaps)) if len(all_gaps) else None,
                        "fraction_under_400ms": float(np.mean(all_gaps < 400))
                        if len(all_gaps)
                        else None,
                    }
                np.savez_compressed(output / "arrays" / f"pair{index:02d}.npz", **arrays)
            results.append(result)
            csv_rows.append(
                {
                    "pair_index": index,
                    "session_token": result["session_token"],
                    "phase": result["phase"],
                    "status": result["status"],
                    "reason": result.get("reason", ""),
                    "offset_seconds": result.get("offset_seconds"),
                    "fit_correlation": (result.get("fit_peak") or {}).get("correlation"),
                    "fit_coverage": result.get("fit_event_coverage"),
                    "validation_correlation": result.get("validation", {}).get("correlation"),
                    "validation_coverage": result.get("validation", {}).get("coverage"),
                    "fit_peak_margin": result.get("fit_peak_margin"),
                    "fit_control_tail_fraction": result.get("fit_control_tail_fraction"),
                    "validation_control_tail_fraction": result.get(
                        "validation_control_tail_fraction"
                    ),
                    "validation_local_residual_ms": result.get("validation_local_residual_ms"),
                    "press_before_audio": result.get("press_before_audio"),
                    "press_after_audio": result.get("press_after_audio"),
                    "provisional_stable_match": result["provisional_stable_match"],
                }
            )
            print(
                f"Timing {index + 1}/54 {result['session_token']} {result['phase']}: {result['status']}, stable={result['provisional_stable_match']}",
                flush=True,
            )
    write_json(output / "pairs.json", {"pairs": results})
    with (output / "timing.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    assessed = [r for r in results if r["status"] == "assessed"]
    stable = [r for r in assessed if r["provisional_stable_match"]]
    by_token = Counter(r["session_token"] for r in stable)
    summary = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": sig,
        "pairs": len(results),
        "assessed_pairs": len(assessed),
        "not_assessed_pairs": len(results) - len(assessed),
        "provisional_stable_pairs": len(stable),
        "tokens_with_both_phases_stable": sorted(k for k, v in by_token.items() if v == 2),
        "failed_criterion_counts": dict(
            Counter(k for r in assessed for k, v in r.get("criteria", {}).items() if not v)
        ),
        "strong_transformed_copy_evidence": copy["strong_transformed_copy_evidence"],
        "models_trained_or_evaluated": 0,
        "segmentation_certified": False,
        "interpretation": "Timing-only diagnostics with log timestamps; provisional flags do not certify event labels or independent microphones/sessions.",
    }
    if signature() != sig:
        raise ResearchError("Masukan/kode berubah selama diagnostik.")
    summary["artifact_sha256"] = {
        str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()
    }
    write_json(output / "summary.json", summary)
    print(f"Completed: {len(stable)}/{len(assessed)} provisional stable pairs", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/skaid-alignment")
    args = parser.parse_args()
    try:
        run(args.output)
    except (ResearchError, OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
