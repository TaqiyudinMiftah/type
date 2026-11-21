"""Lock independent-record supports/queries and decode only their union of real clips."""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .common import (
    ResearchError,
    environment,
    file_hash,
    fresh_directory,
    now,
    read_json,
    write_json,
)
from .skaid import decode, pcm_fingerprint, probe
from .skaid_pilot_common import (
    DATA,
    DRAWS,
    DRIFT,
    KEYS,
    KS,
    NATIVE,
    PROTOCOL,
    ROOT,
    array_hash,
    verify,
)


def signature():
    for base, name in (
        (NATIVE, "summary.json"),
        (DRIFT, "results.json"),
        (ROOT / "reports/skaid-drift-audit", "audit.json"),
    ):
        verify(base, read_json(base / name))
    paths = [
        Path(__file__),
        ROOT / "src/ketik_audio/skaid_pilot_common.py",
        PROTOCOL,
        NATIVE / "summary.json",
        NATIVE / "inventory.json",
        DRIFT / "results.json",
        DRIFT / "label-feasibility.json",
        ROOT / "uv.lock",
    ]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def candidates():
    rows = read_json(DRIFT / "label-feasibility.json")["rows"]
    chosen = next(
        r
        for r in rows
        if r["method"] == "affine_policy"
        and r["length_ms"] == 80
        and r["tier"] == "no_other_key_event"
        and r["vocabulary"] == "exploratory12"
        and r["query_min_per_class"] == 5
        and r["K"] == 10
    )
    tokens = chosen["session_tokens"]
    if len(tokens) != 17 or tokens != sorted(tokens):
        raise ResearchError("Cakupan ID berubah dari audit kelayakan.")
    selected = {r["pair_index"]: r for r in read_json(DRIFT / "selection.json")["records"]}
    inventory = {r["path"]: r for r in read_json(NATIVE / "inventory.json")["files"]}
    result = []
    for row in read_json(DRIFT / "pairs.json")["records"]:
        if row["session_token"] not in tokens:
            continue
        index = row["pair_index"]
        if not row["methods"]["affine_policy"]["provisional_stable_match"]:
            raise ResearchError("Fase peserta belum lolos diagnostik waktu.")
        source = selected[index]["audio"]
        native = inventory[source]
        if native["audio"]["sample_rate"] != 48000 or native["audio"]["channels"] != 1:
            raise ResearchError("Native rate/channel berbeda dari protokol.")
        with np.load(DRIFT / "arrays" / f"pair{index:02d}.npz", allow_pickle=False) as z:
            prefix = "affine_policy_"
            keys = z[f"{prefix}press_key"]
            frames, event_rows = z[f"{prefix}press_audio_frame"], z[f"{prefix}press_row"]
            allowed = z[f"{prefix}w80_no_other_key_event"] & np.isin(keys, KEYS)
            for i in np.flatnonzero(allowed):
                frame, event_row, key = (
                    int(frames[i]),
                    int(event_rows[i]),
                    str(keys[i]),
                )
                start, end = frame - 720, frame + 3120
                if start < 0 or end > native["audio"]["frames"]:
                    raise ResearchError("Crop tidak lengkap.")
                uid = f"{row['session_token']}:{row['phase'][-1]}:{event_row}"
                result.append(
                    {
                        "id": uid,
                        "session_token": row["session_token"],
                        "phase": row["phase"],
                        "row": event_row,
                        "key": key,
                        "label": KEYS.index(key),
                        "pair_index": index,
                        "audio": source,
                        "native_start": start,
                        "native_end": end,
                        "native_pcm_sha256": native["audio"]["pcm_sha256"],
                        "source_file_sha256": native["sha256"],
                    }
                )
    return sorted(result, key=lambda r: (r["session_token"], r["phase"], r["row"]))


def split_plan(pool):
    tokens = sorted({r["session_token"] for r in pool})
    if len(tokens) != 17 or len({r["id"] for r in pool}) != len(pool):
        raise ResearchError("ID/candidate unik tidak sesuai.")
    groups = {
        (t, p, k): [
            r["id"] for r in pool if r["session_token"] == t and r["phase"] == p and r["key"] == k
        ]
        for t in tokens
        for p in ("Phase 1", "Phase 2")
        for k in KEYS
    }
    ranks, queries, needed = {}, {}, set()
    for subject, token in enumerate(tokens):
        rng = np.random.default_rng(20261109 + subject)
        query = []
        for key in KEYS:
            ids = groups[token, "Phase 2", key]
            if len(ids) < 5 or len(groups[token, "Phase 1", key]) < 10:
                raise ResearchError("Candidate tidak cukup untuk K10/query5.")
            query.extend(rng.permutation(ids)[:5].tolist())
        queries[token] = query
        needed.update(query)
        for draw in DRAWS:
            rng = np.random.default_rng(draw + 100000 * subject)
            ranks[token, draw] = {}
            for key in KEYS:
                order = rng.permutation(groups[token, "Phase 1", key]).tolist()
                ranks[token, draw][key] = order
                needed.update(order[:10])
    records = [{**r, "clip_index": i} for i, r in enumerate(r for r in pool if r["id"] in needed)]
    index = {r["id"]: r["clip_index"] for r in records}
    plans = []
    for subject, token in enumerate(tokens):
        for k in KS:
            for draw in DRAWS:
                support = sorted(index[uid] for key in KEYS for uid in ranks[token, draw][key][:k])
                query = sorted(index[uid] for uid in queries[token])
                if len(support) != 12 * k or len(query) != 60 or set(support) & set(query):
                    raise ResearchError("Split atau anggaran tidak valid.")
                support_files = {records[i]["audio"] for i in support}
                query_files = {records[i]["audio"] for i in query}
                if support_files & query_files:
                    raise ResearchError("Kalibrasi/query berbagi rekaman asli.")
                plans.append(
                    {
                        "id": f"s{subject:02d}-k{k}-d{draw}",
                        "subject": subject,
                        "session_token": token,
                        "k": k,
                        "draw": draw,
                        "support": support,
                        "query": query,
                    }
                )
    return records, {
        "keys": KEYS,
        "tokens": tokens,
        "plans": plans,
        "candidate_groups": [
            {"session_token": t, "phase": p, "key": k, "ids": ids}
            for (t, p, k), ids in groups.items()
        ],
        "support_ranks": [
            {"session_token": t, "draw": d, "by_key": r} for (t, d), r in ranks.items()
        ],
    }


def resample_and_pad(native):
    if native.shape != (3840, 1) or not np.isfinite(native).all():
        raise ResearchError("Native crop harus mono48kHz/80ms dan finite.")
    wave = resample_poly(
        native[:, 0].astype(np.float32), 1, 3, window=("kaiser", 5.0), padtype="constant"
    ).astype(np.float32)
    if wave.shape != (1280,):
        raise ResearchError("Panjang resample berbeda.")
    padded = np.zeros(6400, dtype=np.float32)
    padded[560:1840] = wave
    return wave, padded


def prepare(output):
    sig = signature()
    pool = candidates()
    records, plan = split_plan(pool)
    fresh_directory(output)
    write_json(
        output / "lock.json",
        {"created_at": now(), "input_sha256": sig, "environment": environment()},
    )
    write_json(output / "plan.json", {"created_at": now(), **plan})
    (output / "protocol.md").write_bytes(PROTOCOL.read_bytes())
    waves, padded = (
        np.empty((len(records), 1280), np.float32),
        np.empty((len(records), 6400), np.float32),
    )
    seen_native, seen_wave = set(), set()
    inventory = {r["path"]: r for r in read_json(NATIVE / "inventory.json")["files"]}
    files = sorted({r["audio"] for r in records})
    overlap = []
    with (
        zipfile.ZipFile(ROOT / "data/external/skaid-v1/recordings.zip") as archive,
        tempfile.TemporaryDirectory(prefix="skaid-pilot-") as temp,
    ):
        for count, name in enumerate(files, start=1):
            path = Path(temp) / "audio.m4a"
            path.write_bytes(archive.read(name))
            if file_hash(path) != inventory[name]["sha256"]:
                raise ResearchError("Berkas asli berubah.")
            _, stream = probe(path)
            native, rate = decode(path, stream)
            if (
                rate != 48000
                or pcm_fingerprint(native, rate) != inventory[name]["audio"]["pcm_sha256"]
            ):
                raise ResearchError("Native decode tidak cocok dengan audit.")
            selected = [r for r in records if r["audio"] == name]
            starts = np.array([r["native_start"] for r in selected])
            overlap.append(
                {
                    "audio": name,
                    "selected_clips": len(selected),
                    "adjacent_overlapping_crops": int((np.diff(starts) < 3840).sum()),
                }
            )
            for row in selected:
                raw = native[row["native_start"] : row["native_end"]]
                wave, pad = resample_and_pad(raw)
                h_native, h_wave = pcm_fingerprint(raw, rate), array_hash(wave)
                if h_native in seen_native or h_wave in seen_wave:
                    raise ResearchError("Duplikasi PCM persis pada clip terpilih.")
                seen_native.add(h_native)
                seen_wave.add(h_wave)
                i = row["clip_index"]
                waves[i], padded[i] = wave, pad
                row.update(
                    {
                        "crop_native_sha256": h_native,
                        "wave80_sha256": h_wave,
                        "wave400_sha256": array_hash(pad),
                        "rms80": float(np.sqrt(np.mean(wave.astype(float) ** 2))),
                    }
                )
            print(
                f"Prepared native audio {count}/{len(files)}: {len(selected)} selected crops",
                flush=True,
            )
    np.savez_compressed(
        output / "segments.npz",
        wave80=waves,
        wave400=padded,
        y=np.array([r["label"] for r in records], dtype=np.int64),
    )
    write_json(output / "records.json", {"records": records})
    if signature() != sig:
        raise ResearchError("Masukan/kode berubah selama persiapan.")
    result = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": sig,
        "keys": KEYS,
        "subjects": 17,
        "clips": len(records),
        "raw_files": len(files),
        "normal_plans": len(plan["plans"]),
        "query_clips": 1020,
        "sampling_and_segmentation_use_oracle_log_metadata": True,
        "source_capture_overlap_across_support_query": False,
        "within_file_overlap": overlap,
    }
    result["artifact_sha256"] = {
        p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()
    }
    write_json(output / "manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA)
    args = parser.parse_args()
    try:
        prepare(args.output)
    except (ResearchError, OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
