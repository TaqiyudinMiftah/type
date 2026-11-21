"""Audit HEAR and NC1 archives without fitting a key classifier."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

import numpy as np

from ketik_audio.common import (
    LABELS,
    ResearchError,
    environment,
    file_hash,
    fresh_directory,
    now,
    read_json,
    write_json,
)
from ketik_audio.public_recordings import (
    MIN_RMS,
    TIERS,
    VOCABULARIES,
    archive_members,
    cross_record_duplicates,
    hear_crop,
    hear_pairs,
    isolated_window,
    letter,
    nc1_identity,
    parent_error,
    peak_clip80,
    rms,
    sha,
    wav,
)
from ketik_audio.skaid import decode, pcm_fingerprint, probe
from ketik_audio.skaid_pilot_common import KS, ROOT

DATA = ROOT / "data/external"
PROTOCOL = ROOT / "research/PUBLIC_RECORDING_AUDIT_PROTOCOL.md"
HEAR = DATA / "hear-v1"
NC1 = DATA / "nc1-17536616"
HEAR_SHA = "4b0f1908d0d1498ed448577a500acd3a0ad24c0362ba37558034660516b1dcf6"
NC1_SHA = "78a0b8d3d4f6a8e7b3386ccdbe6618037d5ba40aceac30e30c80f5ba32621956"
NC1_CONDITIONS = {
    "1_controlled_conditions_fnatic_gear_rush": "fnatic_quiet",
    "2_background_interference_fnatic_gear_rush": "fnatic_noise",
    "3_controlled_conditions_anne_pro_2": "anne_quiet",
}
TRACKING = "Keystroke_Recordings/tracking.csv"


def signature(kind):
    source = HEAR if kind == "hear" else NC1
    fetch = read_json(source / "fetch.json")
    if fetch["status"] != "complete":
        raise ResearchError("Unduhan belum tervalidasi.")
    if kind == "hear":
        expected = {"HEAR.zip": HEAR_SHA, "release.json": fetch["release_metadata_sha256"]}
    else:
        expected = {name: v["sha256"] for name, v in fetch["files"].items()}
        if expected.get("dataset.zip") != NC1_SHA:
            raise ResearchError("SHA NC1 berbeda dari protokol.")
        expected["record.json"] = fetch["record_sha256"]
    result = {}
    for name, digest in expected.items():
        path = source / name
        if file_hash(path) != digest:
            raise ResearchError(f"Artefak sumber berubah: {name}")
        result[str(path.relative_to(ROOT))] = digest
    paths = [
        source / "fetch.json",
        Path(__file__),
        PROTOCOL,
        ROOT / "src/ketik_audio/public_recordings.py",
        ROOT / "src/ketik_audio/skaid.py",
        ROOT / "src/ketik_audio/skaid_pilot_common.py",
        ROOT / "src/ketik_audio/common.py",
    ]
    result.update({str(p.relative_to(ROOT)): file_hash(p) for p in paths})
    return result


def check_zip(archive):
    files = archive_members(archive)
    if archive.testzip() is not None:
        raise ResearchError("CRC ZIP tidak cocok.")
    return files


def one(members, suffixes):
    result = [r for r in members if PurePosixPath(r.filename).suffix.lower() in suffixes]
    if len(result) != 1:
        raise ResearchError(f"Harus ada tepat satu member dari {suffixes}.")
    return result[0]


def parse_hear_meta(meta, outer_name, tracking):
    participant, device = meta["participant"], meta["device"]
    ident = str(participant["id"]), str(device["deviceId"]), int(participant["session"])
    parts = PurePosixPath(outer_name).parts
    stamp = datetime.fromisoformat(meta["start_time_utc"])
    if (
        len(parts) != 5
        or parts[1] != ident[0]
        or ident[2] < 1
        or stamp.utcoffset() is None
        or len(meta["keys"]) != meta["total_events"]
    ):
        raise ResearchError("Identitas atau log HEAR tidak konsisten.")
    times = np.asarray([float(e["t_ms"]) for e in meta["keys"]], dtype=np.float64)
    duration = float(meta["duration_s"]) * 1000
    rate, channels = int(meta["sample_rate"]), int(meta["num_channels"])
    pre, post = int(meta["pre_ms"]), int(meta["post_ms"])
    if (
        not np.isfinite(times).all()
        or np.any(times < 0)
        or np.any(np.diff(times) < 0)
        or not math.isfinite(duration)
        or duration <= 0
        or rate not in (44100, 48000)
        or channels not in (1, 2)
        or (pre, post) != (60, 200)
    ):
        raise ResearchError("Waktu atau konfigurasi export HEAR berubah.")
    rows = tracking[(*ident, len(times))]
    return ident, stamp.timestamp(), times, duration, rate, channels, pre, post, rows


def hear_export(archive, members, meta, times, parent, parent_rate, index):
    rate, channels = meta["sample_rate"], meta["num_channels"]
    duration, pre, post = 1000 * meta["duration_s"], meta["pre_ms"], meta["post_ms"]
    by_member = {r.filename: r for r in members if r.filename.lower().endswith(".wav")}
    exports = defaultdict(list)
    events = defaultdict(list)
    for n, row in enumerate(meta["keys"]):
        events[(row["key"], times[n])].append(n)
    export_keys = Counter((r["key"], float(r["t_ms"])) for r in meta["exported_files"])
    for n, row in enumerate(meta["exported_files"]):
        exports[row["filename"]].append((n, row))
    diagnostics = Counter()
    diagnostics["export_rows_without_member"] = sum(
        len(v) for k, v in exports.items() if k not in by_member
    )
    diagnostics["export_member_names_with_multiple_rows"] = sum(
        len(v) > 1 for v in exports.values()
    )
    records, formats = [], Counter()
    for member, row in by_member.items():
        blob = archive.read(row)
        linked = exports[member]
        ordinal, declared = linked[0] if len(linked) == 1 else (None, None)
        record = {
            "pack": index,
            "export_index": ordinal,
            "member_sha256": sha(member.encode()),
            "wav_sha256": sha(blob),
            "wav_bytes": len(blob),
            "letter": None,
            "linked": False,
            "isolated80": False,
            "verified80": False,
            "crop_pcm_sha256": None,
            "crop_center_frame": None,
            "reason": None,
        }
        try:
            audio, sample_rate, subtype = wav(blob)
        except ResearchError:
            record["reason"] = "wav_decode"
            records.append(record)
            continue
        diagnostics["wav_decoded"] += 1
        formats[sample_rate, audio.shape[1], subtype] += 1
        if declared is None:
            record["reason"] = "missing_or_ambiguous_export_row"
        else:
            t = float(declared["t_ms"])
            event_key = declared["key"], t
            locations = events[event_key]
            record["letter"] = letter(declared["key"])
            if len(locations) != 1 or export_keys[event_key] != 1:
                record["reason"] = "event_not_unique"
            elif declared["size_bytes"] != len(blob):
                record["reason"] = "size_metadata"
            elif (sample_rate, audio.shape[1]) != (rate, channels):
                record["reason"] = "rate_channel_metadata"
            else:
                record["linked"] = True
                isolated = isolated_window(t, times, locations[0], duration)
                crop, center = hear_crop(audio, rate, t, duration, pre, post)
                if crop is None:
                    record["reason"] = "incomplete_export_or_crop"
                elif not isolated:
                    record["reason"] = "adjacent_keydown"
                elif rms(crop.mean(axis=1, dtype=np.float32)) <= MIN_RMS:
                    record["reason"] = "low_mono_rms"
                else:
                    record["isolated80"] = True
                    record["crop_center_frame"] = center
                    record["crop_pcm_sha256"] = pcm_fingerprint(crop, rate)
                    if (parent_rate, parent.shape[1]) != (rate, channels):
                        record["reason"] = "parent_rate_channel"
                    else:
                        error, delta = parent_error(crop, parent, rate, t)
                        record["parent_relative_error"] = error
                        record["parent_rounding_delta_frames"] = delta
                        record["verified80"] = error is not None and error <= 0.10
                        if not record["verified80"]:
                            record["reason"] = "parent_pcm_error_or_boundary"
        records.append(record)
    return records, diagnostics, formats


def audit_hear(output):
    packs, records = [], []
    formats, wav_formats, exclusions = Counter(), Counter(), Counter()
    participants, devices = {}, {}
    with (
        ZipFile(HEAR / "HEAR.zip") as outer,
        tempfile.TemporaryDirectory(prefix="hear-audit-") as tmp,
    ):
        members = check_zip(outer)
        track = defaultdict(list)
        table = list(csv.DictReader(io.StringIO(outer.read(TRACKING).decode("utf-8-sig"))))
        for i, row in enumerate(table):
            key = row["participantId"], row["deviceId"], int(row["session"]), int(row["totalKeys"])
            track[key].append(i)
        archives = [r for r in members if r.filename.endswith(".zip")]
        if len(archives) != 73 or len(table) != 70:
            raise ResearchError("Coverage paket HEAR berbeda dari inventaris awal.")
        for index, outer_member in enumerate(archives):
            blob = outer.read(outer_member)
            with ZipFile(io.BytesIO(blob)) as inner:
                items = check_zip(inner)
                js = one(items, {".json"})
                media = one(items, {".webm", ".ogg"})
                meta = json.loads(inner.read(js))
                ident, stamp, times, duration, rate, ch, _, _, track_rows = parse_hear_meta(
                    meta, outer_member.filename, track
                )
                participant, device, session = ident
                person = participants.setdefault(participant, f"p{len(participants):02}")
                dev = devices.setdefault(device, f"d{len(devices):02}")
                suffix = PurePosixPath(media.filename).suffix.lower()
                native = Path(tmp) / f"parent{suffix}"
                raw_audio = inner.read(media)
                native.write_bytes(raw_audio)
                _, stream = probe(native)
                parent, native_rate = decode(native, stream)
                rows, diag, wav_types = hear_export(
                    inner, items, meta, times, parent, native_rate, index
                )
                records.extend(rows)
                wav_formats.update(wav_types)
                exclusions.update(diag)
                formats[suffix, stream["codec_name"], native_rate, parent.shape[1]] += 1
                packs.append(
                    {
                        "index": index,
                        "participant": person,
                        "device": dev,
                        "session": session,
                        "package_sha256": sha(blob),
                        "package_member_sha256": sha(outer_member.filename.encode()),
                        "start_relative_seconds": stamp,
                        "track_tuple_match_count": len(track_rows),
                        "track_row": track_rows[0] if len(track_rows) == 1 else None,
                        "event_count": len(times),
                        "export_count": len(meta["exported_files"]),
                        "events_past_duration": int(np.sum(times > duration)),
                        "metadata_rate": rate,
                        "metadata_channels": ch,
                        "metadata_seconds": duration / 1000,
                        "parent_sha256": sha(raw_audio),
                        "parent_pcm_sha256": pcm_fingerprint(parent, native_rate),
                        "native_rate": native_rate,
                        "native_channels": parent.shape[1],
                        "native_frames": len(parent),
                        "native_seconds": len(parent) / native_rate,
                        "container_extension": suffix,
                        "native_codec": stream["codec_name"],
                        "wav_members": len(rows),
                        "wav_decoded": diag["wav_decoded"],
                    }
                )
                del parent, raw_audio
                if (index + 1) % 10 == 0 or index + 1 == len(archives):
                    print(f"HEAR: {index + 1}/{len(archives)} paket diaudit", flush=True)
    origin = min(p["start_relative_seconds"] for p in packs)
    track_use = Counter(p["track_row"] for p in packs if p["track_row"] is not None)
    for p in packs:
        p["start_relative_seconds"] -= origin
        p["track_unique"] = p["track_row"] is not None and track_use[p["track_row"]] == 1
        del p["track_row"]
    duplicates = cross_record_duplicates(records, "crop_pcm_sha256", "pack")
    for r in records:
        if r["crop_pcm_sha256"] in duplicates and r["verified80"]:
            r["reason"] = "cross_package_duplicate_crop"
            r["verified80"] = False
    pairs, feasibility, counts = hear_pairs(packs, records)
    for p in packs:
        p["letters_per_tier"] = {tier: dict(counts[tier][p["index"]]) for tier in TIERS}
    by_person = Counter(p["participant"] for p in packs)
    devices_by_person = defaultdict(set)
    for p in packs:
        devices_by_person[p["participant"]].add(p["device"])
    write_json(output / "packages.json", {"packages": packs})
    write_json(output / "clips.json", {"clips": records})
    write_json(output / "feasibility.json", {"ordered_recording_pairs": pairs, "rows": feasibility})
    result = {
        "source_release": "https://github.com/vikrant-vikram/Hear-Dataset/releases/tag/v1",
        "source_scope": "on-device only",
        "zip_crc_all_members_verified": True,
        "packages": len(packs),
        "participant_ids": len(by_person),
        "participant_ids_with_multiple_packages": sum(n > 1 for n in by_person.values()),
        "participant_ids_with_multiple_device_tokens": sum(
            len(v) > 1 for v in devices_by_person.values()
        ),
        "device_tokens": len(devices),
        "tracking_rows": len(table),
        "tracking_tuple_unique_packages": sum(p["track_unique"] for p in packs),
        "packages_with_events_past_metadata_duration": sum(
            p["events_past_duration"] > 0 for p in packs
        ),
        "events_past_metadata_duration": sum(p["events_past_duration"] for p in packs),
        "packages_with_native_format_matching_metadata": sum(
            (p["native_rate"], p["native_channels"]) == (p["metadata_rate"], p["metadata_channels"])
            for p in packs
        ),
        "duplicate_parent_pcm_groups": sum(
            n > 1 for n in Counter(p["parent_pcm_sha256"] for p in packs).values()
        ),
        "log_events": sum(p["event_count"] for p in packs),
        "export_rows": sum(p["export_count"] for p in packs),
        "wav_members": len(records),
        "wav_decoded": exclusions["wav_decoded"],
        "export_rows_without_member": exclusions["export_rows_without_member"],
        "export_member_names_with_multiple_rows": exclusions[
            "export_member_names_with_multiple_rows"
        ],
        "all_labels_counts_by_tier": {tier: sum(r[tier] for r in records) for tier in TIERS},
        "letters_counts_by_tier": {
            tier: sum(r[tier] and r["letter"] is not None for r in records) for tier in TIERS
        },
        "exclusive_final_clip_status": dict(Counter(r["reason"] or "verified80" for r in records)),
        "cross_package_duplicate_crop_groups": len(duplicates),
        "native_formats": [
            dict(extension=e, codec=c, rate=s, channels=n, count=v)
            for (e, c, s, n), v in sorted(formats.items())
        ],
        "wav_formats": [
            dict(rate=s, channels=n, subtype=t, count=v)
            for (s, n, t), v in sorted(wav_formats.items())
        ],
        "ordered_recording_pairs": len(pairs),
        "feasibility": feasibility,
    }
    return result


def audit_nc1(output):
    records, formats = [], Counter()
    with ZipFile(NC1 / "dataset.zip") as archive:
        members = check_zip(archive)
        if len(members) != 4030:
            raise ResearchError("Coverage NC1 berbeda dari inventaris awal.")
        for index, member in enumerate(members):
            condition, key, stamp = nc1_identity(member.filename)
            group = f"{NC1_CONDITIONS[condition]}:{stamp.date().isoformat()}"
            blob = archive.read(member)
            audio, rate, subtype = wav(blob)
            formats[rate, audio.shape[1], subtype] += 1
            crop, center = peak_clip80(audio, rate)
            level = rms(crop.mean(axis=1, dtype=np.float32)) if crop is not None else None
            records.append(
                {
                    "index": index,
                    "member": member.filename,
                    "condition": NC1_CONDITIONS[condition],
                    "group": group,
                    "date": stamp.date().isoformat(),
                    "timestamp_in_filename": stamp.isoformat(),
                    "zip_date_matches": tuple(member.date_time[:3])
                    == (stamp.year, stamp.month, stamp.day),
                    "letter": key,
                    "rate": rate,
                    "channels": audio.shape[1],
                    "subtype": subtype,
                    "frames": len(audio),
                    "seconds": len(audio) / rate,
                    "peak_center_frame": center,
                    "crop_mono_rms": level,
                    "wav_sha256": sha(blob),
                    "pcm_sha256": pcm_fingerprint(audio, rate),
                    "crop_pcm_sha256": pcm_fingerprint(crop, rate) if crop is not None else None,
                    "eligible80": crop is not None
                    and level > MIN_RMS
                    and rate == 48000
                    and audio.shape[1] == 1,
                }
            )
    full_dups = cross_record_duplicates(records, "pcm_sha256", "group")
    crop_dups = cross_record_duplicates(records, "crop_pcm_sha256", "group")
    unique_stamps = Counter(r["timestamp_in_filename"] for r in records)
    groups, raw_counts = defaultdict(Counter), defaultdict(Counter)
    for r in records:
        r["duplicate_cross_group"] = (
            r["pcm_sha256"] in full_dups or r["crop_pcm_sha256"] in crop_dups
        )
        r["eligible80"] = r["eligible80"] and not r["duplicate_cross_group"]
        r["timestamp_repeated"] = unique_stamps[r["timestamp_in_filename"]] > 1
        raw_counts[r["group"]][r["letter"]] += 1
        if r["eligible80"]:
            groups[r["group"]][r["letter"]] += 1
    details = []
    for group, counts in sorted(raw_counts.items()):
        eligible = groups[group]
        details.append(
            {
                "group": group,
                "wav_count": sum(counts.values()),
                "eligible80": sum(eligible.values()),
                "raw_counts": dict(sorted(counts.items())),
                "eligible_counts": {c: eligible[c] for c in LABELS},
                "missing_letters": [c for c in LABELS if eligible[c] == 0],
            }
        )
    feasibility = []
    for a in sorted(groups):
        for b in sorted(groups):
            if a.split(":")[1] >= b.split(":")[1]:
                continue
            for vocab, keys in VOCABULARIES.items():
                for k in KS:
                    available = [c for c in keys if groups[a][c] >= k and groups[b][c] >= 5]
                    feasibility.append(
                        {
                            "calibration": a,
                            "query": b,
                            "vocabulary": vocab,
                            "k": k,
                            "letters_eligible": available,
                            "all_letters_eligible": len(available) == len(keys),
                        }
                    )
    write_json(output / "clips.json", {"clips": records})
    write_json(output / "feasibility.json", {"groups": details, "rows": feasibility})
    return {
        "source": "https://doi.org/10.5281/zenodo.17536616",
        "license": "CC BY 4.0",
        "zip_crc_all_members_verified": True,
        "wav_members": len(records),
        "wav_decoded": len(records),
        "eligible80": sum(r["eligible80"] for r in records),
        "same_date_zip_filename": sum(r["zip_date_matches"] for r in records),
        "timestamp_duplicate_groups": sum(n > 1 for n in unique_stamps.values()),
        "duplicate_pcm_groups": sum(
            n > 1 for n in Counter(r["pcm_sha256"] for r in records).values()
        ),
        "duplicate_cross_group_full_pcm": len(full_dups),
        "duplicate_cross_group_crop_pcm": len(crop_dups),
        "wav_formats": [
            dict(rate=s, channels=n, subtype=t, count=v) for (s, n, t), v in sorted(formats.items())
        ],
        "condition_day_groups": details,
        "cross_day_pairs_covering_all_by_vocabulary_and_k": {
            vocab: {
                str(k): sum(
                    r["all_letters_eligible"]
                    for r in feasibility
                    if r["vocabulary"] == vocab and r["k"] == k
                )
                for k in KS
            }
            for vocab in VOCABULARIES
        },
    }


def run(kind, output):
    sig = signature(kind)
    fresh_directory(output)
    ffmpeg = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
    write_json(
        output / "lock.json",
        {
            "created_at": now(),
            "input_sha256": sig,
            "environment": environment(),
            "ffmpeg_version": ffmpeg.stdout.splitlines()[0],
            "dataset": kind,
        },
    )
    report = audit_hear(output) if kind == "hear" else audit_nc1(output)
    if signature(kind) != sig:
        raise ResearchError("Masukan berubah selama audit.")
    result = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": sig,
        "dataset": kind,
        **report,
        "artifact_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    write_json(output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("hear", "nc1"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        run(args.dataset, args.output or ROOT / f"reports/{args.dataset}-recording-audit")
    except (ResearchError, OSError, BadZipFile, ValueError) as exc:
        parser.exit(2, f"Audit gagal: {exc}\n")
