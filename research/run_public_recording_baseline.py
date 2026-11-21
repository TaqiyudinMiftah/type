"""Lock held-recording splits, prepare fixed MFCC features, and evaluate Ridge."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

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
from ketik_audio.public_baseline import (
    ALPHA,
    DRAWS,
    check_roles,
    mulaw8k,
    permute,
    prefix,
    ranks,
    scores_metrics,
)
from ketik_audio.public_recordings import (
    archive_members,
    downsample80,
    hear_crop,
    letter,
    peak_clip80,
    sha,
    wav,
)
from ketik_audio.skaid import pcm_fingerprint
from ketik_audio.skaid_pilot_common import CACHE, DATA, KEYS, KS, ROOT, verify
from ketik_audio.skaid_pilot_features import mfcc_features

RUN = ROOT / "runs/public-recording-baseline"
AUDITS = {name: ROOT / f"reports/{name}-recording-audit" for name in ("hear", "nc1")}
HEAR = ROOT / "data/external/hear-v1/HEAR.zip"
NC1 = ROOT / "data/external/nc1-17536616/dataset.zip"
PROTOCOL = ROOT / "research/PUBLIC_RECORDING_BASELINE_PROTOCOL.md"
OLD_SELECTION = ROOT / "runs/skaid-selection"
VOCAB = {"hear": KEYS, "nc1": LABELS}
COMMON_NC1 = tuple(c for c in LABELS if c != "e")
NC1_CALIBRATION = "fnatic_quiet:2025-07-10"
NC1_QUERY = ("fnatic_quiet:2025-07-19", "fnatic_noise:2025-07-19", "anne_quiet:2025-07-19")


def signature():
    paths = [Path(__file__), PROTOCOL, ROOT / "uv.lock"]
    for folder in AUDITS.values():
        p = folder / "summary.json"
        verify(folder, read_json(p))
        paths.append(p)
    for folder, name in (
        (DATA, "manifest.json"),
        (CACHE, "manifest.json"),
        (OLD_SELECTION, "results.json"),
    ):
        p = folder / name
        verify(folder, read_json(p))
        paths.append(p)
    old = read_json(OLD_SELECTION / "selection.json")["choices"]
    mfcc = [c for c in old if c["family"] == "mfcc" and not c["permuted"]]
    if len(mfcc) != 85 or any(
        c["recipe"] != {"normalizer": "row_l2", "alpha": ALPHA} for c in mfcc
    ):
        raise ResearchError("Resep MFCC tidak sama dengan seluruh pilihan SKAID.")
    paths += [
        ROOT / "src/ketik_audio/public_baseline.py",
        ROOT / "src/ketik_audio/public_recordings.py",
        ROOT / "src/ketik_audio/skaid_pilot_features.py",
        ROOT / "src/ketik_audio/skaid_pilot_common.py",
    ]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def hear_groups():
    base = AUDITS["hear"]
    feasible = read_json(base / "feasibility.json")
    packages = {r["index"]: r for r in read_json(base / "packages.json")["packages"]}
    eligible = {
        tuple(p)
        for row in feasible["rows"]
        if row["tier"] == "verified80" and row["vocabulary"] == "skaid12" and row["k"] == 10
        for p in row["pair_ids"]
    }
    found = [
        r for r in feasible["ordered_recording_pairs"] if (r["calibration"], r["query"]) in eligible
    ]

    def time_key(r):
        return (
            packages[r["calibration"]]["start_relative_seconds"],
            packages[r["query"]]["start_relative_seconds"],
            r["calibration"],
            r["query"],
        )

    groups, pairs, seen = set(), [], set()
    for r in sorted(found, key=time_key):
        a, b, person = r["calibration"], r["query"], r["participant"]
        if packages[a]["parent_pcm_sha256"] == packages[b]["parent_pcm_sha256"]:
            raise ResearchError("Audio induk sama pada pasangan HEAR.")
        groups.update([a, b])
        pairs.append(
            {
                "id": f"hear:{a:02d}-{b:02d}",
                "dataset": "hear",
                "subject": person,
                "calibration": f"hear:{a:02d}",
                "query_group": f"hear:{b:02d}",
                "primary": person not in seen,
                "gap_hours": r["gap_hours"],
                "views": [["original", "original"]],
            }
        )
        seen.add(person)
    if (len(pairs), len(seen)) != (18, 3):
        raise ResearchError("Coverage pasangan HEAR tidak sesuai protokol.")
    rows = [
        {
            **r,
            "id": f"h:{r['pack']:02d}:{r['export_index']:04d}",
            "group": f"hear:{r['pack']:02d}",
            "dataset": "hear",
        }
        for r in read_json(base / "clips.json")["clips"]
        if r["pack"] in groups and r["verified80"] and r["letter"] in KEYS
    ]
    return rows, pairs


def nc1_groups():
    groups = {NC1_CALIBRATION, *NC1_QUERY}
    rows = [
        {**r, "dataset": "nc1", "id": f"n:{r['index']:04d}"}
        for r in read_json(AUDITS["nc1"] / "clips.json")["clips"]
        if r["group"] in groups and r["eligible80"]
    ]
    pairs = []
    for i, group in enumerate(NC1_QUERY):
        selected = [r for r in rows if r["group"] == group]
        keys = {r["letter"] for r in selected}
        if len(selected) != (487, 815, 795)[i] or keys != set(COMMON_NC1 if i == 0 else LABELS):
            raise ResearchError("Coverage query NC1 berbeda dari audit.")
        pairs.append(
            {
                "id": f"nc1:{group.split(':')[0]}",
                "dataset": "nc1",
                "subject": "nc1_one_participant",
                "calibration": NC1_CALIBRATION,
                "query_group": group,
                "primary": True,
                "views": [
                    ["original", "original"],
                    ["original", "mulaw8k"],
                    ["mulaw8k", "original"],
                    ["mulaw8k", "mulaw8k"],
                ]
                if i == 0
                else [["original", "original"]],
            }
        )
    return rows, pairs


def make_plan():
    h, hp = hear_groups()
    n, npairs = nc1_groups()
    paths = hp + npairs
    grouped = defaultdict(list)
    for r in h + n:
        grouped[r["group"]].append(r)
    group_keys = {g: i for i, g in enumerate(sorted(grouped))}
    ranks_by_group, queries, needed = {}, {}, set()
    for p in paths:
        a, b = p["calibration"], p["query_group"]
        keys = VOCAB[p["dataset"]]
        if a not in ranks_by_group:
            orders = {str(d): ranks(grouped[a], keys, 10, group_keys[a], d, 1) for d in DRAWS}
            ranks_by_group[a] = orders
            needed.update(i for rank in orders.values() for i in prefix(rank, keys, 10))
        if b not in queries:
            if p["dataset"] == "hear":
                orders = ranks(grouped[b], keys, 5, group_keys[b], 0, 2)
                queries[b] = prefix(orders, keys, 5)
            else:
                queries[b] = sorted(r["id"] for r in grouped[b])
            needed.update(queries[b])
    records = sorted((r for r in h + n if r["id"] in needed), key=lambda r: r["id"])
    check_roles(records, paths, ranks_by_group, queries, VOCAB)
    return {
        "records": records,
        "paths": paths,
        "rankings": ranks_by_group,
        "queries": queries,
        "group_indices": group_keys,
        "eligible_pool_counts": {g: len(rs) for g, rs in sorted(grouped.items())},
        "draws": DRAWS,
        "k": KS,
        "vocabularies": VOCAB,
    }


def clip_wave(archive, r, meta=None):
    if meta is None:
        name = r["member"]
    else:
        row = meta["exported_files"][r["export_index"]]
        name = row["filename"]
        if sha(name.encode()) != r["member_sha256"] or letter(row["key"]) != r["letter"]:
            raise ResearchError("Referensi clip HEAR berbeda dari audit.")
    blob = archive.read(name)
    if sha(blob) != r["wav_sha256"]:
        raise ResearchError("SHA WAV berbeda dari audit.")
    audio, rate, _ = wav(blob)
    if meta is None:
        crop, center = peak_clip80(audio, rate)
        expected = r["peak_center_frame"]
    else:
        crop, center = hear_crop(
            audio, rate, row["t_ms"], 1000 * meta["duration_s"], meta["pre_ms"], meta["post_ms"]
        )
        expected = r["crop_center_frame"]
    if crop is None or center != expected or pcm_fingerprint(crop, rate) != r["crop_pcm_sha256"]:
        raise ResearchError("Crop tidak mereproduksi audit eksplisit.")
    return downsample80(crop, rate)


def extract_waves(plan):
    records = plan["records"]
    values = np.empty((len(records), 1280), dtype=np.float32)
    filled = set()
    by_pack = defaultdict(list)
    for i, r in enumerate(records):
        if r["dataset"] == "hear":
            by_pack[r["pack"]].append((i, r))
    packs = {r["index"]: r for r in read_json(AUDITS["hear"] / "packages.json")["packages"]}
    with ZipFile(HEAR) as outer:
        archives = [r for r in archive_members(outer) if r.filename.endswith(".zip")]
        for pack, rows in sorted(by_pack.items()):
            raw = outer.read(archives[pack])
            if sha(raw) != packs[pack]["package_sha256"]:
                raise ResearchError("Arsip sesi berbeda dari audit.")
            with ZipFile(io.BytesIO(raw)) as inner:
                (js,) = [r for r in archive_members(inner) if r.filename.endswith(".json")]
                meta = json.loads(inner.read(js))
                for i, r in rows:
                    values[i] = clip_wave(inner, r, meta)
                    filled.add(i)
    with ZipFile(NC1) as archive:
        for i, r in enumerate(records):
            if r["dataset"] == "nc1":
                values[i] = clip_wave(archive, r)
                filled.add(i)
    if len(filled) != len(records) or not np.isfinite(values).all():
        raise ResearchError("Coverage waveform tidak lengkap.")
    return values


def ska_source():
    records = read_json(DATA / "records.json")["records"]
    indices = np.array([i for i, r in enumerate(records) if r["phase"] == "Phase 1"])
    if len(records) != 6137 or len(indices) != 5117:
        raise ResearchError("Label sumber SKAID tidak sama dengan protokol.")
    with np.load(CACHE / "features.npz", allow_pickle=False) as z:
        x = z["mfcc"][indices]
    labels = np.asarray([KEYS.index(records[i]["key"]) for i in indices])
    return x, labels, indices


def prepare(output):
    sig = signature()
    fresh_directory(output)
    plan = make_plan()
    write_json(output / "plan.json", plan)
    write_json(
        output / "plan-lock.json",
        {
            "created_at": now(),
            "input_sha256": sig,
            "alpha": ALPHA,
            "plan_sha256": file_hash(output / "plan.json"),
            "environment": environment(),
        },
    )
    (output / "protocol.md").write_bytes(PROTOCOL.read_bytes())
    waves = extract_waves(plan)
    codec_indices = np.array(
        [i for i, r in enumerate(plan["records"]) if r["group"] in (NC1_CALIBRATION, NC1_QUERY[0])]
    )
    codec_waves = mulaw8k(waves[codec_indices])
    np.savez_compressed(
        output / "waves.npz", original=waves, mulaw8k=codec_waves, codec_indices=codec_indices
    )
    with threadpool_limits(limits=1):
        orig, codec = mfcc_features(waves), mfcc_features(codec_waves)
    x, y, indices = ska_source()
    np.savez_compressed(
        output / "features.npz",
        original=orig,
        mulaw8k=codec,
        codec_indices=codec_indices,
        source_features=x,
        source_labels=y,
        source_indices=indices,
    )
    record = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": sig,
        "n_waveforms": len(waves),
        "n_codec_waveforms": len(codec_waves),
        "n_pairs": len(plan["paths"]),
        "ffmpeg_version": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[
            0
        ],
        "artifact_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    if signature() != sig:
        raise ResearchError("Masukan berubah selama persiapan.")
    write_json(output / "prepared.json", record)
    print(f"Plan terkunci: {len(waves)} klip, {len(codec_waves)} versi codec", flush=True)
    return record


def head_fit(x, y, keys):
    if set(y.tolist()) != set(range(len(keys))):
        raise ResearchError("Dukungan tidak mencakup semua kelas classifier.")
    x = normalize(x.astype(np.float64))
    head = RidgeClassifier(
        alpha=ALPHA, solver="cholesky", fit_intercept=True, class_weight=None
    ).fit(x, y)
    if not np.array_equal(head.classes_, np.arange(len(keys))):
        raise ResearchError("Urutan output kelas berubah.")
    return head, float(np.mean(head.predict(x) == y))


def feature_lookup(z):
    x = z["original"]
    codec = np.full_like(x, np.nan)
    codec[z["codec_indices"]] = z["mulaw8k"]
    return {"original": x, "mulaw8k": codec}


def evaluate(output):
    prepared = read_json(output / "prepared.json")
    verify(output, prepared)
    if prepared["input_sha256"] != signature() or (output / "results.json").exists():
        raise ResearchError("Masukan berubah atau evaluasi sudah selesai.")
    plan = read_json(output / "plan.json")
    lock = read_json(output / "plan-lock.json")
    if file_hash(output / "plan.json") != lock["plan_sha256"] or lock["alpha"] != ALPHA:
        raise ResearchError("Lock plan tidak sesuai.")
    check_roles(plan["records"], plan["paths"], plan["rankings"], plan["queries"], VOCAB)
    start = now()
    ids = {r["id"]: i for i, r in enumerate(plan["records"])}
    labels = np.array([VOCAB[r["dataset"]].index(r["letter"]) for r in plan["records"]])
    fits, parameters, trials, query_arrays = {}, {}, [], {}
    with np.load(output / "features.npz", allow_pickle=False) as z, threadpool_limits(limits=1):
        views = feature_lookup(z)
        for p in plan["paths"]:
            a, b, keys = p["calibration"], p["query_group"], VOCAB[p["dataset"]]
            query = [ids[i] for i in plan["queries"][b]]
            conditions = [(k, d, False) for k in KS for d in DRAWS] + [(10, d, True) for d in DRAWS]
            if p["dataset"] == "hear":
                conditions = [(0, None, False)] + conditions
            for cview, qview in p["views"]:
                q = normalize(views[qview][query].astype(np.float64))
                for k, d, control in conditions:
                    split = (
                        (a, cview, k, d, control)
                        if k
                        else ("skaid_source", "original", 0, None, False)
                    )
                    if split not in fits:
                        s = (
                            [ids[i] for i in prefix(plan["rankings"][a][str(d)], keys, k)]
                            if k
                            else []
                        )
                        sx, sy = (
                            (views[cview][s], labels[s])
                            if k
                            else (z["source_features"], z["source_labels"])
                        )
                        y = permute(sy, plan["group_indices"][a], d) if control else sy
                        head, train_accuracy = head_fit(sx, y, keys)
                        hid = f"head{len(fits):03d}"
                        fits[split] = (head, hid, train_accuracy, s, y.tolist())
                        parameters[f"{hid}_coef"] = head.coef_
                        parameters[f"{hid}_intercept"] = head.intercept_
                    head, hid, train_accuracy, s, y = fits[split]
                    scores = head.decision_function(q)
                    tid = f"trial{len(trials):03d}"
                    query_arrays[tid] = scores
                    trials.append(
                        {
                            **p,
                            "trial_id": tid,
                            "head_id": hid,
                            "calibration_view": cview,
                            "query_view": qview,
                            "k": k,
                            "draw": d,
                            "permuted": control,
                            "support_indices": s,
                            "query_indices": query,
                            "fit_labels": y if k else None,
                            "fit_support_accuracy": train_accuracy,
                            "target_fit_labels": len(s),
                            "source_fit_labels": 0 if k else 5117,
                            "query": scores_metrics(labels[query], scores, keys),
                            "query_common25": scores_metrics(
                                labels[query], scores, keys, COMMON_NC1
                            )
                            if p["dataset"] == "nc1"
                            else None,
                        }
                    )
    if len(trials) != 450 or any(not np.isfinite(v).all() for v in parameters.values()):
        raise ResearchError("Coverage hasil/head tidak sesuai protokol.")
    np.savez_compressed(output / "heads.npz", **parameters)
    np.savez_compressed(output / "scores.npz", **query_arrays)
    write_json(output / "trials.json", {"trials": trials})
    verify(output, prepared)
    result = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": prepared["input_sha256"],
        "evaluation_started_at": start,
        "plan_sha256": lock["plan_sha256"],
        "alpha": ALPHA,
        "fits": len(fits),
        "trials": len(trials),
        "predictions_repeated": sum(len(v) for v in query_arrays.values()),
        "target_validation_labels": 0,
        "new_encoder_extractions": 0,
        "artifact_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    write_json(output / "results.json", result)
    print(f"Evaluasi selesai: {len(trials)} kondisi, {len(fits)} fit unik", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "evaluate"))
    parser.add_argument("--output", type=Path, default=RUN)
    args = parser.parse_args()
    try:
        {"prepare": prepare, "evaluate": evaluate}[args.stage](args.output)
    except (OSError, ResearchError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
