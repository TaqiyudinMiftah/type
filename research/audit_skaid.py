"""Verify public SKAID audio, logs, native clocks and explicit phase pairing on CPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
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
from ketik_audio.skaid import (
    PHASES,
    audio_details,
    audit_log,
    decode,
    envelopes,
    probe,
    recording_identity,
    session_token,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/external/skaid-v1"
CATALOG = ROOT / "data/external/skaid-catalog"
PROTOCOL = ROOT / "research/SKAID_AUDIT_PROTOCOL.md"
PINS = {
    "data/external/skaid-v1/recordings.zip": (
        "1e669e2734e504823df86781093cd5f85f8817af6bdb4bae48db6bc2886f3b4a"
    ),
    "data/external/skaid-v1/demographics.csv": (
        "5dcc02ad315d01f91563a9434a856d198b9f52a1d26036b940cd332e0f047569"
    ),
    "data/external/skaid-catalog/logs.zip": (
        "5e84c19209df891b88e8de600f171511e9af370be2de903d85353bd0b2c19eda"
    ),
    "data/external/skaid-catalog/README.txt": (
        "c51f24e74879a79f36d37e3da56ce7e79a23a5b83b227a5b2948c66f960b93e3"
    ),
}


def signature():
    values = {name: file_hash(ROOT / name) for name in PINS}
    if values != PINS:
        raise ResearchError("Berkas sumber SKAID berbeda dari pin.")
    fetch = read_json(DATA / "fetch.json")
    if fetch["status"] != "complete" or fetch["source_metadata_sha256"] != file_hash(
        DATA / "source.json"
    ):
        raise ResearchError("Unduhan belum lengkap/metadata berubah.")
    if fetch["fetch_code_sha256"] != file_hash(ROOT / "research/fetch_skaid.py"):
        raise ResearchError("Kode unduhan berubah dari provenance.")
    paths = [
        PROTOCOL,
        Path(__file__),
        ROOT / "src/ketik_audio/skaid.py",
        ROOT / "research/fetch_skaid.py",
        DATA / "fetch.json",
        DATA / "source.json",
        CATALOG / "record.json",
        CATALOG / "fetch.json",
    ]
    values.update({str(p.relative_to(ROOT)): file_hash(p) for p in paths})
    return values


def metadata_times(probed):
    values = []
    for scope, entry in [
        ("format", probed.get("format", {})),
        *[(f"stream:{s['index']}", s) for s in probed.get("streams", [])],
    ]:
        for key, value in entry.get("tags", {}).items():
            if "time" not in key.lower() and "date" not in key.lower():
                continue
            record = {"scope": scope, "tag": key, "value": value, "epoch_ms": None}
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if parsed.tzinfo is not None:
                    record["epoch_ms"] = round(parsed.timestamp() * 1000)
                    record["timezone_present"] = True
                else:
                    record["timezone_present"] = False
            except ValueError:
                record["timezone_present"] = None
            values.append(record)
    return values


def duplicate_sets(rows, field):
    groups = defaultdict(list)
    for row in rows:
        value = row.get("audio", {}).get(field) if field == "pcm_sha256" else row.get(field)
        if value:
            groups[value].append(row["path"])
    return [{"digest": key, "paths": paths} for key, paths in groups.items() if len(paths) > 1]


def run(output):
    sig = signature()
    fresh_directory(output)
    versions = {
        name: subprocess.run(
            [name, "-version"], capture_output=True, text=True, check=True
        ).stdout.splitlines()[0]
        for name in ("ffmpeg", "ffprobe")
    }
    write_json(
        output / "lock.json", {"created_at": now(), "input_sha256": sig, "decoders": versions}
    )
    (output / "protocol.md").write_bytes(PROTOCOL.read_bytes())
    logs, packed = [], {}
    with zipfile.ZipFile(CATALOG / "logs.zip") as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".csv") or name.endswith("_full_text.csv"):
                continue
            detail, events = audit_log(archive.read(name), name)
            detail["event_array_prefix"] = f"log{len(logs):02d}"
            for field in ("row", "timestamp_ms", "phase", "event", "raw_key", "physical_key"):
                packed[f"{detail['event_array_prefix']}_{field}"] = np.asarray(
                    [e[field] for e in events]
                )
            logs.append(detail)
    write_json(output / "logs.json", {"logs": logs})
    np.savez_compressed(output / "events.npz", **packed)
    with (DATA / "demographics.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        demographics = list(reader)
    if set(reader.fieldnames or ()) != {
        "Session ID",
        "Age Range",
        "Typing Method",
        "Device Used",
        "Email",
    }:
        raise ResearchError("Kolom demografi berubah.")
    by_demographic = defaultdict(list)
    for row in demographics:
        by_demographic[session_token(row["Session ID"])].append(row)
    inventory, env_arrays = [], {}
    with zipfile.ZipFile(DATA / "recordings.zip") as archive:
        if len(set(archive.namelist())) != len(archive.namelist()):
            raise ResearchError("ZIP memiliki nama anggota berulang.")
        for info in archive.infolist():
            if info.is_dir():
                continue
            blob = archive.read(info)  # Full read verifies ZIP CRC.
            identity = recording_identity(info.filename)
            row = {
                "path": info.filename,
                "bytes": info.file_size,
                "crc32": f"{info.CRC:08x}",
                "sha256": hashlib.sha256(blob).hexdigest(),
                "identity": identity,
                "decode_status": "not_audio",
            }
            if identity:
                index = len(inventory)
                with tempfile.TemporaryDirectory(prefix="skaid-native-") as temporary:
                    path = Path(temporary) / f"audio.{identity['extension']}"
                    path.write_bytes(blob)
                    try:
                        probed, stream = probe(path)
                        write_json(output / "probes" / f"audio{index:02d}.json", probed)
                        row["probe"] = f"probes/audio{index:02d}.json"
                        row["codec_name"] = stream.get("codec_name")
                        row["container_duration_seconds"] = float(
                            probed.get("format", {}).get("duration", 0)
                        )
                        row["metadata_times"] = metadata_times(probed)
                        audio, rate = decode(path, stream)
                        row["audio"] = audio_details(audio, rate)
                        edges, rms, peak = envelopes(audio, rate)
                        row["envelope_prefix"] = f"audio{index:02d}"
                        for key, values in (("edges_native", edges), ("rms", rms), ("peak", peak)):
                            env_arrays[f"audio{index:02d}_{key}"] = values
                        row["decode_status"] = "decoded"
                    except (ResearchError, subprocess.TimeoutExpired) as exc:
                        row["decode_status"] = "failed"
                        row["decode_error"] = str(exc)
            inventory.append(row)
            print(
                f"Audited {len(inventory)} archive files: {row['decode_status']} {info.filename}",
                flush=True,
            )
    write_json(output / "inventory.json", {"files": inventory})
    np.savez_compressed(output / "envelopes.npz", **env_arrays)
    pairing = []
    used = set()
    for log in logs:
        tokens = log["session_tokens"]
        token = tokens[0] if len(tokens) == 1 and tokens == log["filename_tokens"] else None
        for phase in PHASES:
            candidates = [
                row
                for row in inventory
                if row["identity"]
                and row["identity"]["session_token"] == token
                and row["identity"]["phase"] == phase
            ]
            used.update(row["path"] for row in candidates)
            anchors = []
            first_stamp = log["phases"][phase]["min_timestamp_ms"]
            for row in candidates:
                for timing in row.get("metadata_times", []):
                    if timing["epoch_ms"] is not None:
                        anchors.append(
                            {
                                "path": row["path"],
                                **timing,
                                "first_log_minus_tag_seconds": (
                                    (first_stamp - timing["epoch_ms"]) / 1000
                                    if first_stamp is not None
                                    else None
                                ),
                            }
                        )
            span = log["phases"][phase]["span_ms"] / 1000
            pairing.append(
                {
                    "log": log["file"],
                    "session_token": token,
                    "phase": phase,
                    "log_phase": log["phases"][phase],
                    "candidates": [r["path"] for r in candidates],
                    "canonical_m4a_candidates": [
                        r["path"] for r in candidates if r["identity"]["extension"] == "m4a"
                    ],
                    "matching_demographic_rows": len(by_demographic.get(token, [])),
                    "timestamp_clock_usable": log["timestamp_unique_count"] > 1
                    and not log["backwards_timestamp_rows"],
                    "candidate_audio_minus_log_span_seconds": {
                        r["path"]: r["audio"]["duration_seconds"] - span
                        for r in candidates
                        if "audio" in r
                    },
                    "metadata_anchor_candidates": anchors,
                    "audio_event_alignment_verified": False,
                }
            )
    duplicates = {
        "file": duplicate_sets(inventory, "sha256"),
        "pcm": duplicate_sets(inventory, "pcm_sha256"),
    }
    write_json(output / "duplicates.json", duplicates)
    write_json(
        output / "pairing.json",
        {
            "pairs": pairing,
            "unmatched_recordings": [
                r["path"] for r in inventory if r["identity"] and r["path"] not in used
            ],
        },
    )
    summary = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": sig,
        "source": "https://zenodo.org/records/17282184",
        "license": "CC BY 4.0",
        "environment": environment(),
        "decoders": versions,
        "archive_files_crc_checked": len(inventory),
        "logs": len(logs),
        "log_events": sum(r["events"] for r in logs),
        "decoded_audio": sum(r["decode_status"] == "decoded" for r in inventory),
        "decode_failures": sum(r["decode_status"] == "failed" for r in inventory),
        "canonical_m4a": sum(
            bool(r["identity"] and r["identity"]["extension"] == "m4a") for r in inventory
        ),
        "raw_audio_pcm_unique": len({r["audio"]["pcm_sha256"] for r in inventory if "audio" in r}),
        "exact_duplicate_pcm_groups": len(duplicates["pcm"]),
        "missing_audio_pairs": [
            {"session_token": p["session_token"], "phase": p["phase"]}
            for p in pairing
            if not p["candidates"]
        ],
        "multiple_audio_candidates": [
            {"session_token": p["session_token"], "phase": p["phase"]}
            for p in pairing
            if len(p["candidates"]) > 1
        ],
        "collapsed_log_tokens": [
            r["session_tokens"] for r in logs if r["timestamp_unique_count"] == 1
        ],
        "demographics_rows": len(demographics),
        "device_category_counts": dict(Counter(r["Device Used"] for r in demographics)),
        "sample_rate_counts": dict(
            Counter(str(r["audio"]["sample_rate"]) for r in inventory if "audio" in r)
        ),
        "channels_counts": dict(
            Counter(str(r["audio"]["channels"]) for r in inventory if "audio" in r)
        ),
        "total_decoded_seconds": sum(
            r["audio"]["duration_seconds"] for r in inventory if "audio" in r
        ),
        "models_trained_or_evaluated": 0,
        "audio_event_alignment_verified": False,
        "eligibility": "pairing inventory only; offset/drift and event crops not verified",
    }
    if signature() != sig:
        raise ResearchError("Input/kode berubah selama audit.")
    summary["artifact_sha256"] = {
        str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()
    }
    write_json(output / "summary.json", summary)
    print(
        f"SKAID native audio audit complete: {summary['decoded_audio']} decoded files", flush=True
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/skaid-audit")
    args = parser.parse_args()
    try:
        run(args.output)
    except (ResearchError, OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
