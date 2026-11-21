"""Build a deduplicated 36-key manifest without inventing session splits."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, read_json, write_json
from ketik_audio.mka import KEYS, PLATFORMS

ROOT = Path(__file__).resolve().parents[1]


def run(audit, headers, output, snapshot):
    summary = read_json(audit / "summary.json")
    if summary["status"] != "complete":
        raise ResearchError("Audit MKA belum selesai.")
    for name, digest in summary["artifact_sha256"].items():
        if file_hash(audit / name) != digest:
            raise ResearchError(f"Artefak audit berubah: {name}")
    rows = read_json(audit / "inventory.json")["files"]
    traces = read_json(audit / "excerpt-provenance.json")["traces"]
    header_check = read_json(headers)
    if header_check["inventory_sha256"] != file_hash(audit / "inventory.json"):
        raise ResearchError("Header dan inventaris tidak cocok.")
    by_path = {r["path"]: r for r in rows}
    by_pcm = defaultdict(list)
    for row in rows:
        if "audio" in row:
            by_pcm[row["audio"]["pcm_sha256"]].append(row["path"])
    groups, unique_clips = [], set()
    for platform in PLATFORMS:
        for key in KEYS:
            raw = [
                r
                for r in rows
                if r["scope"] == "canonical"
                and r["platform"] == platform
                and r["role"] == "raw"
                and r["label"] == key
            ]
            if len(raw) != 1:
                raise ResearchError("Induk nominal tidak tunggal.")
            parent = raw[0]
            selected = [t for t in traces if t["platform"] == platform and t["key"] == key]
            unique = {}
            for trace in sorted(selected, key=lambda r: (len(r["path"]), r["path"])):
                row = by_path[trace["path"]]
                audio = row["audio"]
                if not trace["matched"] or not trace.get("all_channels_pcm_equal"):
                    raise ResearchError(
                        "Potongan belum terverifikasi sebagai cuplikan penuh induknya."
                    )
                if (
                    trace["parent"] != parent["path"]
                    or trace["stop_frame"] - trace["start_frame"] != audio["frames"]
                ):
                    raise ResearchError("Frame/induk potongan tidak konsisten.")
                if audio["pcm_sha256"] in unique:
                    continue
                unique_clips.add(audio["pcm_sha256"])
                unique[audio["pcm_sha256"]] = {
                    "path": trace["path"],
                    "sha256": row["sha256"],
                    "pcm_sha256": audio["pcm_sha256"],
                    "start_frame": trace["start_frame"],
                    "stop_frame": trace["stop_frame"],
                    "duration_seconds": audio["duration_seconds"],
                    "exact_pcm_aliases": by_pcm[audio["pcm_sha256"]],
                }
            groups.append(
                {
                    "platform_folder": platform,
                    "key": key,
                    "parent_group_id": f"pcm-sha256:{parent['audio']['pcm_sha256']}",
                    "raw_path": parent["path"],
                    "raw_sha256": parent["sha256"],
                    "raw_pcm_aliases": by_pcm[parent["audio"]["pcm_sha256"]],
                    "sample_rate": parent["audio"]["sample_rate"],
                    "channels": parent["audio"]["channels"],
                    "session_id": None,
                    "keyboard_id": None,
                    "participant_id": None,
                    "assigned_split": None,
                    "supplied_unique_clips": list(unique.values()),
                }
            )
    result = {
        "created_at": now(),
        "source": summary["source"],
        "creators": summary["creators"],
        "license": summary["license"],
        "archive_sha256": summary["archive_sha256"],
        "audit_summary_sha256": file_hash(audit / "summary.json"),
        "header_check_sha256": file_hash(headers),
        "curation_code_sha256": file_hash(Path(__file__)),
        "class_order": KEYS,
        "unit": "supplied audio excerpt, not an annotated isolated keystroke",
        "label_source": "author file/folder names",
        "parent_groups": len(groups),
        "unique_parent_pcm": len({g["parent_group_id"] for g in groups}),
        "unique_supplied_clip_pcm": len(unique_clips),
        "minimum_unique_clips_per_group": min(len(g["supplied_unique_clips"]) for g in groups),
        "cross_recording_calibration_test_eligible": False,
        "split_decision": summary["decision"],
        "groups": groups,
    }
    if snapshot and snapshot.exists():
        raise ResearchError("Snapshot kurasi sudah ada.")
    fresh_directory(output)
    write_json(output / "manifest.json", result)
    compact = {k: v for k, v in result.items() if k != "groups"}
    compact["manifest_sha256"] = file_hash(output / "manifest.json")
    write_json(output / "summary.json", compact)
    if snapshot:
        write_json(snapshot, compact)
    print(
        f"Curated {len(groups)} parent groups, {len(unique_clips)} unique supplied clips; "
        "no session split assigned"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=ROOT / "reports/mka-audit")
    parser.add_argument("--headers", type=Path, default=ROOT / "reports/mka-headers/headers.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/mka-36")
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    try:
        run(args.audit, args.headers, args.output, args.snapshot)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
