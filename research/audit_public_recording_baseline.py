"""Re-decode the selected clips and replay fixed-Ridge scores independently."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from run_public_recording_baseline import (
    COMMON_NC1,
    NC1_CALIBRATION,
    NC1_QUERY,
    RUN,
    VOCAB,
    extract_waves,
    feature_lookup,
    make_plan,
    ska_source,
)
from threadpoolctl import threadpool_limits

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, read_json, write_json
from ketik_audio.public_baseline import (
    ALPHA,
    DRAWS,
    check_roles,
    mulaw8k,
    permute,
    prefix,
    scores_metrics,
)
from ketik_audio.skaid_pilot_common import KS, ROOT, verify
from ketik_audio.skaid_pilot_features import mfcc_features

OUTPUT = ROOT / "reports/public-recording-baseline-audit"


def l2(v):
    x = v.astype(np.float64)
    norm = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
    return x / np.where(norm > 10 * np.finfo(np.float64).eps, norm, 1)


def direct_ridge(x, y, nclasses):
    x = l2(x)
    target = 2 * np.eye(nclasses)[y] - 1
    mean, tm = x.mean(0), target.mean(0)
    z = x - mean
    w = np.linalg.solve(z.T @ z + ALPHA * np.eye(z.shape[1]), z.T @ (target - tm))
    return w.T, tm - mean @ w


def expected_trials(plan):
    conditions = [(k, d, False) for k in KS for d in DRAWS] + [(10, d, True) for d in DRAWS]
    return {
        (p["id"], a, b, k, d, ctrl)
        for p in plan["paths"]
        for a, b in p["views"]
        for k, d, ctrl in (conditions + ([(0, None, False)] if p["dataset"] == "hear" else []))
    }


def verify_features(plan):
    waves = extract_waves(plan)
    codec = np.array(
        [i for i, r in enumerate(plan["records"]) if r["group"] in (NC1_CALIBRATION, NC1_QUERY[0])]
    )
    coded = mulaw8k(waves[codec])
    with np.load(RUN / "waves.npz", allow_pickle=False) as z:
        for key, expected in (("original", waves), ("mulaw8k", coded), ("codec_indices", codec)):
            if not np.array_equal(z[key], expected):
                raise ResearchError(f"Waveform terpilih berbeda pada re-decode: {key}")
    x, y, idx = ska_source()
    with np.load(RUN / "features.npz", allow_pickle=False) as z:
        for key, expected in (
            ("original", mfcc_features(waves)),
            ("mulaw8k", mfcc_features(coded)),
            ("codec_indices", codec),
            ("source_features", x),
            ("source_labels", y),
            ("source_indices", idx),
        ):
            if not np.array_equal(z[key], expected):
                raise ResearchError(f"Replay fitur tidak identik: {key}")
    return len(waves), len(coded)


def run(output):
    result = read_json(RUN / "results.json")
    verify(RUN, result)
    plan = read_json(RUN / "plan.json")
    lock = read_json(RUN / "plan-lock.json")
    if not lock["created_at"] < result["evaluation_started_at"] < result["created_at"]:
        raise ResearchError("Urutan lock dan hasil tidak sesuai.")
    if plan != json.loads(json.dumps(make_plan())):
        raise ResearchError("Ranking/split tak bisa direproduksi dari audit.")
    check_roles(plan["records"], plan["paths"], plan["rankings"], plan["queries"], VOCAB)
    paths = {p["id"]: p for p in plan["paths"]}
    trials = read_json(RUN / "trials.json")["trials"]
    seen = {
        (t["id"], t["calibration_view"], t["query_view"], t["k"], t["draw"], t["permuted"])
        for t in trials
    }
    if seen != expected_trials(plan) or len(trials) != len(seen):
        raise ResearchError("Cakupan evaluasi/set kontrol tidak lengkap.")
    fresh_directory(output)
    sig = {str(p.relative_to(ROOT)): file_hash(p) for p in (Path(__file__), RUN / "results.json")}
    write_json(output / "lock.json", {"created_at": now(), "input_sha256": sig})
    checks, maxima, changes = [], Counter(), Counter()
    source_labels = read_json(RUN / "plan.json")["records"]
    ids = {r["id"]: i for i, r in enumerate(source_labels)}
    labels = np.array([VOCAB[r["dataset"]].index(r["letter"]) for r in source_labels])
    with (
        threadpool_limits(limits=1),
        np.load(RUN / "features.npz", allow_pickle=False) as z,
        np.load(RUN / "heads.npz", allow_pickle=False) as models,
        np.load(RUN / "scores.npz", allow_pickle=False) as saved_scores,
        (output / "classifications.csv").open("w", encoding="utf-8", newline="") as stream,
    ):
        waves, codec = verify_features(plan)
        features = feature_lookup(z)
        x0, y0 = z["source_features"], z["source_labels"]
        writer = csv.writer(stream)
        writer.writerow(
            [
                "trial",
                "pair",
                "dataset",
                "k",
                "draw",
                "control",
                "calibration_view",
                "query_view",
                "clip",
                "true",
                "prediction",
                "replay",
            ]
        )
        replays = {}
        for t in trials:
            p, k, draw, ctrl = paths[t["id"]], t["k"], t["draw"], t["permuted"]
            tid, hid = t["trial_id"], t["head_id"]
            keys = VOCAB[p["dataset"]]
            a, b = p["calibration"], p["query_group"]
            si, qi = t["support_indices"], t["query_indices"]
            target_count = len(keys) * k
            if (
                qi != [ids[i] for i in plan["queries"][b]]
                or len(si) != target_count
                or t["target_fit_labels"] != target_count
                or t["source_fit_labels"] != (0 if k else 5117)
            ):
                raise ResearchError("Label budget atau query berbeda dari lock.")
            if k:
                order = prefix(plan["rankings"][a][str(draw)], keys, k)
                if si != [ids[i] for i in order] or set(si) & set(qi):
                    raise ResearchError("Dukungan berbeda dari lock atau tumpang tindih query.")
                x, y = features[t["calibration_view"]][si], labels[si]
                if ctrl:
                    y = permute(y, plan["group_indices"][a], draw)
                if y.tolist() != t["fit_labels"] or not np.array_equal(
                    np.bincount(y), np.full(len(keys), k)
                ):
                    raise ResearchError("Label fit target atau kontrol tidak sesuai.")
            else:
                x, y = x0, y0
                if t["fit_labels"] is not None:
                    raise ResearchError("K0 tidak boleh menerima fit label HEAR.")
            head_key = (a, t["calibration_view"], k, draw, ctrl) if k else ("SKAID-K0",)
            if hid not in replays:
                w, bias = direct_ridge(x, y, len(keys))
                replays[hid] = head_key, w, bias
            locked, w, bias = replays[hid]
            if locked != head_key:
                raise ResearchError("Satu kepala dipakai untuk dukungan yang berbeda.")
            orig, inter = models[f"{hid}_coef"], models[f"{hid}_intercept"]
            q = l2(features[t["query_view"]][qi])
            recorded = saved_scores[tid]
            replay = q @ w.T + bias
            direct_orig = q @ orig.T + inter
            for name, left, right in (
                ("coef", w, orig),
                ("intercept", bias, inter),
                ("scores", replay, recorded),
                ("serialize", direct_orig, recorded),
            ):
                err = float(np.max(np.abs(left - right)))
                maxima[name] = max(maxima[name], err)
                if not np.allclose(left, right, rtol=1e-8, atol=1e-8):
                    raise ResearchError(f"Replay {name} di luar toleransi: {err}.")
            if scores_metrics(labels[qi], recorded, keys) != t["query"] or (
                p["dataset"] == "nc1"
                and scores_metrics(labels[qi], recorded, keys, COMMON_NC1) != t["query_common25"]
            ):
                raise ResearchError("Metrik query berbeda dari skor tersimpan.")
            if not np.isclose(
                np.mean((l2(x) @ orig.T + inter).argmax(1) == y), t["fit_support_accuracy"]
            ):
                raise ResearchError("Metrik dukungan berbeda dari parameter.")
            prediction, predicted = recorded.argmax(1), replay.argmax(1)
            changes[p["dataset"]] += int(np.sum(prediction != predicted))
            checks.append(
                {
                    "trial": tid,
                    "replay_correct": int(np.sum(predicted == labels[qi])),
                    "prediction_changes": int(np.sum(prediction != predicted)),
                    "max_abs_score_error": float(np.max(np.abs(replay - recorded))),
                }
            )
            for j, i in enumerate(qi):
                writer.writerow(
                    [
                        tid,
                        p["id"],
                        p["dataset"],
                        k,
                        draw,
                        ctrl,
                        t["calibration_view"],
                        t["query_view"],
                        source_labels[i]["id"],
                        keys[labels[i]],
                        keys[prediction[j]],
                        keys[predicted[j]],
                    ]
                )
        if len(replays) != result["fits"]:
            raise ResearchError("Replay kepala model tidak lengkap.")
    write_json(output / "trial-checks.json", {"checks": checks})
    verify(RUN, result)
    if any(file_hash(ROOT / p) != h for p, h in sig.items()):
        raise ResearchError("Masukan audit berubah.")
    record = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": sig,
        "waveforms_redecoded": waves,
        "codec_recomputed": codec,
        "heads_replayed": len(replays),
        "trials_replayed": len(checks),
        "max_abs_differences": dict(maxima),
        "prediction_changes": dict(changes),
        "recording_split_and_budget_verified": True,
        "artifact_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    write_json(output / "audit.json", record)
    print(f"Audit selesai: {len(checks)} kondisi, {waves} crop dan {codec} versi codec", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        run(args.output)
    except (OSError, ResearchError, ValueError) as exc:
        parser.exit(2, f"Audit gagal: {exc}\n")
