"""Audit native Praat Sound objects stored among MKA feature files."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from collections import defaultdict
from pathlib import Path

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, read_json, write_json
from ketik_audio.mka import pcm_digest
from ketik_audio.praat_sound import read_praat_sound

ROOT = Path(__file__).resolve().parents[1]


def run(data, audit, output, snapshot):
    summary = read_json(audit / "summary.json")
    if summary["status"] != "complete":
        raise ResearchError("Audit WAV belum selesai.")
    for name, digest in summary["artifact_sha256"].items():
        if file_hash(audit / name) != digest:
            raise ResearchError("Artefak audit WAV berubah.")
    archive = data / "MKA datasets.zip"
    if file_hash(archive) != summary["archive_sha256"]:
        raise ResearchError("Arsip berbeda dari audit WAV.")
    if snapshot and snapshot.exists():
        raise ResearchError("Snapshot Praat sudah ada.")
    fresh_directory(output)
    inventory = read_json(audit / "inventory.json")["files"]
    wav = defaultdict(list)
    for row in inventory:
        if "audio" in row:
            wav[row["audio"]["pcm_sha256"]].append(row["path"])
    decoded, checked = {}, []
    selected = [r for r in inventory if r.get("text", {}).get("object_class") == "Sound 2"]
    with zipfile.ZipFile(archive) as source:
        for number, row in enumerate(selected, 1):
            blob = source.read(row["path"])
            if hashlib.sha256(blob).hexdigest() != row["sha256"]:
                raise ResearchError("Isi objek Praat berbeda dari inventaris.")
            if row["sha256"] not in decoded:
                audio, rate = read_praat_sound(blob)
                digest = pcm_digest(audio, rate)
                decoded[row["sha256"]] = {
                    "sample_rate": rate,
                    "frames": len(audio),
                    "channels": audio.shape[1],
                    "pcm_sha256": digest,
                    "exact_decoded_waveform_matches": wav.get(digest, []),
                }
            checked.append({"path": row["path"], "scope": row["scope"], **decoded[row["sha256"]]})
            if number % 20 == 0:
                print(f"Praat {number}/{len(selected)} objects decoded", flush=True)
    additional = {r["pcm_sha256"] for r in checked if not r["exact_decoded_waveform_matches"]}
    result = {
        "created_at": now(),
        "status": "complete",
        "source": summary["source"],
        "creators": summary["creators"],
        "license": summary["license"],
        "archive_sha256": summary["archive_sha256"],
        "wav_audit_sha256": file_hash(audit / "summary.json"),
        "decoder_code_sha256": file_hash(ROOT / "src/ketik_audio/praat_sound.py"),
        "audit_code_sha256": file_hash(Path(__file__)),
        "objects_decoded": len(checked),
        "canonical_objects": sum(r["scope"] == "canonical" for r in checked),
        "aggregate_objects": sum(r["scope"] == "aggregate" for r in checked),
        "objects_with_exact_existing_waveform_match": sum(
            bool(r["exact_decoded_waveform_matches"]) for r in checked
        ),
        "additional_unique_pcm": len(additional),
        "files": checked,
    }
    write_json(output / "objects.json", result)
    compact = {k: v for k, v in result.items() if k != "files"}
    compact["objects_sha256"] = file_hash(output / "objects.json")
    write_json(output / "summary.json", compact)
    if snapshot:
        write_json(snapshot, compact)
    print(f"COMPLETE: {len(checked)} Sound objects; {len(additional)} additional PCM waveforms")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/external/mka-v3")
    parser.add_argument("--audit", type=Path, default=ROOT / "reports/mka-audit")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/mka-praat-audit")
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    try:
        run(args.data, args.audit, args.output, args.snapshot)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
