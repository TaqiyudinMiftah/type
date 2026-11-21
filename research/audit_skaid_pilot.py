"""Replay every SKAID crop, embedding and linear probe with independent CPU checks."""

from __future__ import annotations

import argparse
import csv
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, read_json, write_json
from ketik_audio.skaid import decode, pcm_fingerprint, probe
from ketik_audio.skaid_pilot_common import (
    CACHE,
    CHECKPOINT,
    DATA,
    ENCODERS,
    KEYS,
    NATIVE,
    ROOT,
    RUN,
    SEEDS,
    SOURCE,
    array_hash,
    verify,
)
from ketik_audio.skaid_pilot_data import candidates, resample_and_pad, split_plan
from ketik_audio.skaid_pilot_features import mfcc_features
from ketik_audio.skaid_pilot_probe import (
    feature_rows,
    fit_head,
    metrics,
    permuted_labels,
    validate_plan,
)


def data_replay(records, wave80, wave400):
    base_records, plan = split_plan(candidates())
    saved_plan = read_json(DATA / "plan.json")
    for key, value in plan.items():
        expected = list(value) if key == "keys" else value
        if saved_plan[key] != expected:
            raise ResearchError("Regenerasi split/rank berbeda.")
    if len(base_records) != len(records):
        raise ResearchError("Jumlah crop berbeda.")
    for base, saved in zip(base_records, records, strict=True):
        if any(saved[k] != v for k, v in base.items()):
            raise ResearchError("Identitas/proyeksi crop berbeda.")
    checks = []
    with (
        zipfile.ZipFile(ROOT / "data/external/skaid-v1/recordings.zip") as archive,
        tempfile.TemporaryDirectory(prefix="skaid-pilot-replay-") as temp,
    ):
        for name in sorted({r["audio"] for r in records}):
            selected = [r for r in records if r["audio"] == name]
            path = Path(temp) / "audio.m4a"
            path.write_bytes(archive.read(name))
            if file_hash(path) != selected[0]["source_file_sha256"]:
                raise ResearchError("Berkas audio sumber berbeda.")
            _, stream = probe(path)
            native, rate = decode(path, stream)
            if pcm_fingerprint(native, rate) != selected[0]["native_pcm_sha256"]:
                raise ResearchError("Decode PCM native berbeda.")
            for row in selected:
                i = row["clip_index"]
                raw = native[row["native_start"] : row["native_end"]]
                wave, pad = resample_and_pad(raw)
                if (
                    pcm_fingerprint(raw, rate) != row["crop_native_sha256"]
                    or array_hash(wave) != row["wave80_sha256"]
                    or array_hash(pad) != row["wave400_sha256"]
                ):
                    raise ResearchError("Hash crop/resample/padding tidak cocok.")
                if not np.array_equal(wave, wave80[i]) or not np.array_equal(pad, wave400[i]):
                    raise ResearchError("Array crop tidak identik pada replay.")
            checks.append({"audio": name, "clips_redecoded": len(selected), "all_exact": True})
            print(f"Replayed raw audio {len(checks)}/34", flush=True)
    return checks


def cpu_features(wave80, wave400, primary_metadata):
    import torch

    from ketik_audio.fewshot import _beats, frontend
    from ketik_audio.frozen import extract, frozen_encoder
    from ketik_audio.mka_lopo import extract_beats
    from ketik_audio.resnet_fewshot import initialize_model, state_digest

    torch.set_num_threads(4)
    arrays, states = {"mfcc": mfcc_features(wave80)}, []
    images = []
    with torch.inference_mode():
        for offset in range(0, len(wave400), 64):
            images.append(frontend(torch.from_numpy(wave400[offset : offset + 64])).numpy())
    arrays["mel64"] = np.concatenate(images)
    del images
    model, _ = _beats(CHECKPOINT, "cpu")
    before = state_digest(model.state_dict())
    arrays["beats"] = extract_beats(model, wave400, "cpu", progress=True)
    after = state_digest(model.state_dict())
    if before != after or before != primary_metadata["beats"]["before"]:
        raise ResearchError("State BEATs CPU tidak cocok/tidak beku.")
    states.append({"encoder": "beats", "before": before, "after": after, "device": "cpu"})
    del model
    for kind in ("pretrained", "random"):
        for seed in SEEDS:
            folder = SOURCE / f"sources/s{seed}"
            name = f"{kind}_s{seed}"
            state = (
                torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
                if kind == "pretrained"
                else None
            )
            model = initialize_model(seed, state).requires_grad_(False).eval()
            if kind == "random":
                saved = torch.load(CACHE / f"{name}.pt", map_location="cpu", weights_only=True)
                if state_digest(saved) != state_digest(model.state_dict()):
                    raise ResearchError("Inisialisasi random tidak dapat direproduksi.")
            before = state_digest(model.state_dict())
            encoder = frozen_encoder(model)
            with np.load(folder / "normalization.npz", allow_pickle=False) as z:
                mean, std = z["mean"], z["std"]
            arrays[name] = extract(encoder, arrays["mel64"], mean, std, "cpu")
            if kind == "pretrained":
                with torch.inference_mode():
                    arrays[f"source_logits_s{seed}"] = model[-1](
                        torch.from_numpy(arrays[name])
                    ).numpy()
            after = state_digest(model.state_dict())
            if before != after or before != primary_metadata[name]["before"]:
                raise ResearchError("State ResNet CPU berbeda/berubah.")
            states.append({"encoder": name, "before": before, "after": after, "device": "cpu"})
            print(f"CPU extracted {name}: {len(wave400)} embeddings", flush=True)
            del encoder, model
    return arrays, states


def dual_ridge(x, labels):
    target = 2 * np.eye(12)[labels] - 1
    xc, yc = x - x.mean(0), target - target.mean(0)
    dual = np.linalg.solve(xc @ xc.T + np.eye(len(x)), yc)
    coefficient = dual.T @ xc
    intercept = target.mean(0) - coefficient @ x.mean(0)
    return coefficient, intercept


def run(output):
    manifests = [
        (NATIVE, read_json(NATIVE / "summary.json")),
        (DATA, read_json(DATA / "manifest.json")),
        (CACHE, read_json(CACHE / "manifest.json")),
        (RUN, read_json(RUN / "results.json")),
    ]
    for base, record in manifests:
        verify(base, record)
    fresh_directory(output)
    paths = [Path(__file__), DATA / "manifest.json", CACHE / "manifest.json", RUN / "results.json"]
    sig = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    write_json(output / "lock.json", {"created_at": now(), "input_sha256": sig, "device": "cpu"})
    with np.load(DATA / "segments.npz", allow_pickle=False) as z:
        wave80, wave400, labels = z["wave80"], z["wave400"], z["y"]
    records = read_json(DATA / "records.json")["records"]
    plans = read_json(DATA / "plan.json")["plans"]
    validate_plan(records, plans, labels)
    crops = data_replay(records, wave80, wave400)
    write_json(output / "crop-checks.json", {"checks": crops})
    metadata = {r["encoder"]: r for r in manifests[2][1]["encoders"]}
    cpu, states = cpu_features(wave80, wave400, metadata)
    np.savez_compressed(output / "cpu-features.npz", **cpu)
    conditions = [{**p, "permuted": False} for p in plans] + [
        {**p, "permuted": True} for p in plans if p["k"] == 10
    ]
    saved_trials = read_json(RUN / "trials.json")
    by_trial = {(r["encoder"], r["array_index"]): r for r in saved_trials["trials"]}
    feature_checks, head_checks = [], []
    maximum_direct_error, maximum_replay_error, cpu_disagreements = 0.0, 0.0, 0
    csv_path = output / "classifications.csv"
    with (
        csv_path.open("w", newline="", encoding="utf-8") as stream,
        np.load(CACHE / "features.npz", allow_pickle=False) as primary,
    ):
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "trial_id",
                "encoder",
                "subject",
                "k",
                "draw",
                "permuted",
                "clip_id",
                "true_key",
                "primary_prediction",
                "cpu_prediction",
            ],
        )
        writer.writeheader()
        for name, values in cpu.items():
            expected = primary[name]
            if values.shape != expected.shape or not np.allclose(
                values, expected, atol=1e-3, rtol=1e-3
            ):
                raise ResearchError(f"Fitur CPU melewati toleransi: {name}")
            feature_checks.append(
                {
                    "name": name,
                    "shape": list(values.shape),
                    "max_abs_error": float(np.abs(values - expected).max()),
                    "within_atol_rtol_1e3": True,
                }
            )
        write_json(output / "feature-checks.json", {"checks": feature_checks, "states": states})
        for encoder in ENCODERS:
            x, cx = feature_rows(primary[encoder]), feature_rows(cpu[encoder])
            cpu_scores = []
            with np.load(RUN / f"{encoder}.npz", allow_pickle=False) as saved:
                for i, p in enumerate(conditions):
                    s, q = p["support"], p["query"]
                    fit_y = permuted_labels(labels, p) if p["permuted"] else labels[s]
                    trial = by_trial[encoder, i]
                    if (
                        not np.array_equal(saved["support"][i, : len(s)], s)
                        or not np.array_equal(saved["query"][i], q)
                        or not np.array_equal(saved["fit_labels"][i, : len(s)], fit_y)
                    ):
                        raise ResearchError("Indeks/fit-label kepala berbeda dari protokol.")
                    head, scores, support_scores, _ = fit_head(x, labels, s, q, fit_y)
                    for actual, expected in (
                        (head.coef_, saved["coef"][i]),
                        (head.intercept_, saved["intercept"][i]),
                        (scores, saved["query_scores"][i]),
                        (support_scores, saved["support_scores"][i, : len(s)]),
                    ):
                        error = float(np.max(np.abs(actual - expected)))
                        maximum_replay_error = max(maximum_replay_error, error)
                        if error > 1e-8:
                            raise ResearchError("Replay kepala/skor cache berbeda.")
                    coefficient, intercept = dual_ridge(x[s], fit_y)
                    for actual, expected in (
                        (coefficient, saved["coef"][i]),
                        (intercept, saved["intercept"][i]),
                        (x[q] @ coefficient.T + intercept, saved["query_scores"][i]),
                    ):
                        error = float(np.max(np.abs(actual - expected)))
                        maximum_direct_error = max(maximum_direct_error, error)
                        if error > 1e-8:
                            raise ResearchError("Dual ridge independen berbeda.")
                    if metrics(labels[q], scores) != trial["query"]:
                        raise ResearchError("Metrik utama tidak cocok pada replay.")
                    if (
                        float(np.mean(support_scores.argmax(1) == fit_y))
                        != trial["support_fit_label_accuracy"]
                        or float(np.mean(support_scores.argmax(1) == labels[s]))
                        != trial["support_true_label_accuracy"]
                    ):
                        raise ResearchError("Metrik dukungan berbeda.")
                    _, c_scores, _, _ = fit_head(cx, labels, s, q, fit_y)
                    cpu_scores.append(c_scores)
                    flips = int(np.sum(scores.argmax(1) != c_scores.argmax(1)))
                    cpu_disagreements += flips
                    head_checks.append(
                        {
                            "id": trial["id"],
                            "query_replay": trial["query"],
                            "query_cpu_features": metrics(labels[q], c_scores),
                            "cpu_feature_prediction_disagreements": flips,
                            "max_cpu_feature_score_error": float(np.abs(c_scores - scores).max()),
                        }
                    )
                    for j, clip in enumerate(q):
                        writer.writerow(
                            {
                                "trial_id": trial["id"],
                                "encoder": encoder,
                                "subject": p["subject"],
                                "k": p["k"],
                                "draw": p["draw"],
                                "permuted": p["permuted"],
                                "clip_id": records[clip]["id"],
                                "true_key": KEYS[int(labels[clip])],
                                "primary_prediction": KEYS[int(scores[j].argmax())],
                                "cpu_prediction": KEYS[int(c_scores[j].argmax())],
                            }
                        )
            np.savez_compressed(
                output / f"cpu-scores-{encoder}.npz", query_scores=np.stack(cpu_scores)
            )
            print(f"Audited {encoder}:306 heads against replay/dual solve/CPU features", flush=True)
        from ketik_audio.harrison import KEYS as source_keys

        columns = [source_keys.index(k) for k in KEYS]
        zero_checks = []
        with np.load(RUN / "zero-shot.npz", allow_pickle=False) as saved:
            for i, trial in enumerate(saved_trials["zero_shot"]):
                q = saved["query"][i]
                seed = trial["source_seed"]
                scores = primary[f"source_logits_s{seed}"][q][:, columns]
                c_scores = cpu[f"source_logits_s{seed}"][q][:, columns]
                if (
                    not np.array_equal(scores, saved["scores"][i])
                    or metrics(labels[q], scores) != trial["query"]
                ):
                    raise ResearchError("K0 source-head replay berbeda.")
                flips = int(np.sum(scores.argmax(1) != c_scores.argmax(1)))
                zero_checks.append(
                    {
                        "subject": trial["subject"],
                        "seed": seed,
                        "primary": trial["query"],
                        "cpu": metrics(labels[q], c_scores),
                        "prediction_disagreements": flips,
                    }
                )
                for j, clip in enumerate(q):
                    writer.writerow(
                        {
                            "trial_id": f"zero-s{seed}-subject{trial['subject']}",
                            "encoder": "zero_shot_resnet",
                            "subject": trial["subject"],
                            "k": 0,
                            "draw": "",
                            "permuted": False,
                            "clip_id": records[clip]["id"],
                            "true_key": KEYS[int(labels[clip])],
                            "primary_prediction": KEYS[int(scores[j].argmax())],
                            "cpu_prediction": KEYS[int(c_scores[j].argmax())],
                        }
                    )
    write_json(output / "head-checks.json", {"heads": head_checks, "zero_shot": zero_checks})
    for base, record in manifests:
        verify(base, record)
    for name, digest in sig.items():
        if file_hash(ROOT / name) != digest:
            raise ResearchError("Masukan/kode audit berubah.")
    result = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": sig,
        "native_files_redecoded": len(crops),
        "clips_replayed_exact": len(records),
        "all_encoder_embeddings_recomputed_cpu": len(records),
        "heads_replayed": len(head_checks),
        "zero_shot_replayed": len(zero_checks),
        "primary_metrics_replay_match": True,
        "max_head_replay_error": maximum_replay_error,
        "max_independent_dual_error": maximum_direct_error,
        "cpu_feature_query_prediction_disagreements_repeated_trials": cpu_disagreements,
        "zero_shot_cpu_prediction_disagreements_repeated_trials": sum(
            t["prediction_disagreements"] for t in zero_checks
        ),
        "feature_checks": feature_checks,
        "all_neural_states_match_and_unchanged": True,
    }
    result["artifact_sha256"] = {
        p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()
    }
    write_json(output / "audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/skaid-pilot-audit")
    args = parser.parse_args()
    try:
        run(args.output)
    except (ResearchError, OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
