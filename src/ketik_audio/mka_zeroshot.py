"""Locked, source-only classification of supplied MKA excerpts on CPU."""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import time
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .common import ResearchError, environment, file_hash, now, read_json, write_json
from .fewshot import PRE_SAMPLES, RATE, ROOT, SAMPLES, _score, frontend
from .harrison import measure
from .mka import KEYS, PLATFORMS, pcm_digest
from .resnet_fewshot import initialize_model, state_digest

PROTOCOL = ROOT / "research/MKA_ZEROSHOT_PROTOCOL.md"
ARCHIVE = ROOT / "data/external/mka-v3/MKA datasets.zip"
MANIFEST = ROOT / "reports/mka-36/manifest.json"
SOURCE = ROOT / "runs/resnet-fewshot"
SEEDS = (42, 43, 44)
POLICIES = ("peak", "tiled")
MANIFEST_SHA = "fc8697c58742521e0db5274bfd609d44b54dbc579b8083d719e7909cf6d87e05"
SOURCE_RESULTS_SHA = "f519b4e93e74392f7407d67f1c45e4574a3ffab0893359ddf05f5462cf974178"


def array_digest(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256(str((values.shape, str(values.dtype))).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def resample_mono(audio, rate):
    """Same channel mean, precision and rational resampling as source preparation."""
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all() or rate <= 0:
        raise ResearchError("Waveform/rate tidak valid.")
    mono = values.mean(axis=1)
    factor = math.gcd(int(rate), RATE)
    return resample_poly(mono, RATE // factor, int(rate) // factor).astype(np.float32)


def crop_layout(wave):
    """Label-independent peak crop and full-coverage tiled sensitivity."""
    if wave.ndim != 1 or len(wave) < SAMPLES or not np.isfinite(wave).all():
        raise ResearchError("Audio harus finite dan sedikitnya 400 ms.")
    peak = int(np.abs(wave).argmax())
    start = min(max(peak - PRE_SAMPLES, 0), len(wave) - SAMPLES)
    tiled = list(range(0, len(wave) - SAMPLES + 1, SAMPLES // 2))
    if tiled[-1] != len(wave) - SAMPLES:
        tiled.append(len(wave) - SAMPLES)
    return {
        "resampled_frames": len(wave),
        "peak_frame": peak,
        "peak_start": start,
        "peak_clamped": start != peak - PRE_SAMPLES,
        "tiled_starts": tiled,
    }


def grouped_mean(scores, owners, count):
    """Arithmetic logit mean; groups need not be contiguous or equally sized."""
    owners = np.asarray(owners)
    if (
        scores.ndim != 2
        or len(scores) != len(owners)
        or owners.dtype.kind not in "iu"
        or len(owners) == 0
        or owners.min() < 0
        or owners.max() >= count
        or not np.isfinite(scores).all()
    ):
        raise ResearchError("Skor/pemilik kelompok tidak valid.")
    counts = np.bincount(owners, minlength=count)
    if (counts == 0).any():
        raise ResearchError("Ada kelompok tanpa skor.")
    result = np.zeros((count, scores.shape[1]), dtype=np.float64)
    np.add.at(result, owners, scores)
    return result / counts[:, None]


def input_paths():
    paths = [
        ARCHIVE,
        MANIFEST,
        PROTOCOL,
        Path(__file__),
        ROOT / "reports/mka-audit/summary.json",
        ROOT / "research/curate_mka.py",
        ROOT / "src/ketik_audio/mka.py",
        ROOT / "src/ketik_audio/fewshot.py",
        ROOT / "src/ketik_audio/harrison.py",
        ROOT / "src/ketik_audio/cnn_repair.py",
        ROOT / "src/ketik_audio/resnet_fewshot.py",
        ROOT / "research/RESNET_FEWSHOT_PROTOCOL.md",
        ROOT / "data/external/harrison-fewshot/segments.npz",
        ROOT / "data/external/harrison-fewshot-features/features.npz",
        SOURCE / "sources-locked.json",
        SOURCE / "results.json",
    ]
    for seed in SEEDS:
        paths.extend(
            SOURCE / f"sources/s{seed}" / name
            for name in ("best.pt", "normalization.npz", "config.json", "result.json", "scores.npz")
        )
        paths.extend(
            SOURCE / f"conditions/b{SEEDS.index(seed) + 1}-k0" / name
            for name in ("config.json", "result.json", "scores.npz")
        )
    return paths


def signature():
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in input_paths()}
    manifest = read_json(MANIFEST)
    old_lock = read_json(SOURCE / "sources-locked.json")
    if hashes[str(MANIFEST.relative_to(ROOT))] != MANIFEST_SHA:
        raise ResearchError("Manifest kurasi berubah.")
    if hashes[str(ARCHIVE.relative_to(ROOT))] != manifest["archive_sha256"]:
        raise ResearchError("Arsip MKA berubah.")
    if hashes[str((SOURCE / "results.json").relative_to(ROOT))] != SOURCE_RESULTS_SHA:
        raise ResearchError("Hasil sumber/kurva lama berubah.")
    fields = {
        "model_implementation_sha256": "src/ketik_audio/cnn_repair.py",
        "fewshot_implementation_sha256": "src/ketik_audio/fewshot.py",
        "harrison_implementation_sha256": "src/ketik_audio/harrison.py",
        "implementation_sha256": "src/ketik_audio/resnet_fewshot.py",
        "protocol_sha256": "research/RESNET_FEWSHOT_PROTOCOL.md",
        "segments_sha256": "data/external/harrison-fewshot/segments.npz",
        "features_sha256": "data/external/harrison-fewshot-features/features.npz",
    }
    for name, path in fields.items():
        if hashes[path] != old_lock["signature"][name]:
            raise ResearchError(f"Input sumber berubah: {path}")
    for seed in SEEDS:
        prefix = f"runs/resnet-fewshot/sources/s{seed}"
        result = read_json(ROOT / prefix / "result.json")
        if (
            hashes[f"{prefix}/best.pt"] != old_lock["source_checkpoints"][str(seed)]
            or hashes[f"{prefix}/normalization.npz"] != result["normalization_sha256"]
        ):
            raise ResearchError("Checkpoint/statistik sumber berubah.")
    return hashes


def extract_inputs():
    """Read only curated unique excerpts; preserve full provenance in row order."""
    manifest = read_json(MANIFEST)
    if manifest["class_order"] != KEYS or len(manifest["groups"]) != 216:
        raise ResearchError("Skema manifest berbeda dari protokol.")
    records, peaks, tiles, owners = [], [], [], []
    seen_pcm, seen_groups = set(), set()
    with zipfile.ZipFile(ARCHIVE) as archive:
        for group_index, group in enumerate(manifest["groups"]):
            identity = (group["platform_folder"], group["key"])
            if identity in seen_groups or len(group["supplied_unique_clips"]) != 6:
                raise ResearchError("Kelompok berulang atau jumlah potongan bukan enam.")
            seen_groups.add(identity)
            for clip in group["supplied_unique_clips"]:
                blob = archive.read(clip["path"])
                audio, rate = sf.read(io.BytesIO(blob), dtype="float64", always_2d=True)
                digest = pcm_digest(audio, rate)
                if (
                    hashlib.sha256(blob).hexdigest() != clip["sha256"]
                    or digest != clip["pcm_sha256"]
                    or digest in seen_pcm
                    or rate != group["sample_rate"]
                    or audio.shape[1] != group["channels"]
                ):
                    raise ResearchError(f"Identitas potongan berbeda: {clip['path']}")
                seen_pcm.add(digest)
                mono = resample_mono(audio, rate)
                layout = crop_layout(mono)
                index = len(records)
                peaks.append(mono[layout["peak_start"] : layout["peak_start"] + SAMPLES])
                for start in layout["tiled_starts"]:
                    tiles.append(mono[start : start + SAMPLES])
                    owners.append(index)
                records.append(
                    {
                        "clip_index": index,
                        "group_index": group_index,
                        "platform": group["platform_folder"],
                        "key": group["key"],
                        "label": KEYS.index(group["key"]),
                        "parent_group_id": group["parent_group_id"],
                        "raw_path": group["raw_path"],
                        **clip,
                        **layout,
                        "mono_sha256": array_digest(mono),
                    }
                )
    if len(records) != 1296 or seen_groups != {(p, k) for p in PLATFORMS for k in KEYS}:
        raise ResearchError("Cakupan 36 kelas/enam platform tidak lengkap.")
    return records, {
        "peak_wave": np.stack(peaks),
        "tiled_wave": np.stack(tiles),
        "tiled_owner": np.asarray(owners, dtype=np.int64),
        "clip_group": np.asarray([r["group_index"] for r in records], dtype=np.int64),
        "y": np.asarray([r["label"] for r in records], dtype=np.int64),
    }


def mel_cpu(waves):
    import torch

    with torch.inference_mode():
        return np.concatenate(
            [
                frontend(torch.from_numpy(waves[i : i + 32])).numpy()
                for i in range(0, len(waves), 32)
            ]
        )


def metrics(y, scores):
    result = measure(y, scores)
    result.pop("wilson95")  # Clips share source WAVs; no iid binomial interval.
    histogram = np.bincount(scores.argmax(1), minlength=len(KEYS))
    result.update(
        {
            "predicted_class_count": int((histogram > 0).sum()),
            "prediction_histogram": histogram.tolist(),
            "most_frequent_prediction": KEYS[int(histogram.argmax())],
            "most_frequent_prediction_count": int(histogram.max()),
        }
    )
    return result


def measurements(records, score_arrays):
    rows = []
    y = np.asarray([r["label"] for r in records])
    group = np.asarray([r["group_index"] for r in records])
    parents = [next(r for r in records if r["group_index"] == i) for i in range(216)]
    parent_y = np.asarray([r["label"] for r in parents])
    for seed in SEEDS:
        for policy in POLICIES:
            scores = score_arrays[f"s{seed}_{policy}_clip"]
            grouped = grouped_mean(scores, group, 216)
            for platform in PLATFORMS:
                ix = np.asarray([r["platform"] == platform for r in records])
                pix = np.asarray([r["platform"] == platform for r in parents])
                rows.append(
                    {
                        "seed": seed,
                        "policy": policy,
                        "platform": platform,
                        "clip": metrics(y[ix], scores[ix]),
                        "parent": metrics(parent_y[pix], grouped[pix]),
                    }
                )
    groups = []
    for policy in POLICIES:
        for platform in PLATFORMS:
            selected = [r for r in rows if r["policy"] == policy and r["platform"] == platform]
            summary = {"policy": policy, "platform": platform}
            for unit in ("clip", "parent"):
                summary[unit] = {}
                for name in ("accuracy", "top5", "macro_f1"):
                    values = np.asarray([r[unit][name] for r in selected])
                    summary[unit][name] = {
                        "mean": float(values.mean()),
                        "sd": float(values.std(ddof=1)),
                        "per_seed": values.tolist(),
                    }
            groups.append(summary)
    return {"conditions": rows, "groups": groups}


def controls(seed, model, mean, std, source_data, source_mel, output):
    """Replay old source holdout and target K=0; no new target fitting."""
    results = []
    controls = (("zoom", SOURCE / f"sources/s{seed}/scores.npz"),)
    controls += (("phone", SOURCE / f"conditions/b{SEEDS.index(seed) + 1}-k0/scores.npz"),)
    for domain, old_path in controls:
        with np.load(old_path, allow_pickle=False) as saved:
            indices, old_scores, y = saved["indices"], saved["scores"], saved["y"]
        if not np.array_equal(y, source_data[f"{domain}_y"][indices]):
            raise ResearchError("Label kontrol lama berubah.")
        mel = source_mel[domain][indices]
        fresh_mel = mel_cpu(source_data[f"{domain}_wave"][indices])
        scores = _score(model, mel, mean, std, "cpu")
        fresh_scores = _score(model, fresh_mel, mean, std, "cpu")
        if (
            not np.array_equal(scores.argmax(1), old_scores.argmax(1))
            or not np.array_equal(fresh_scores.argmax(1), old_scores.argmax(1))
            or measure(y, scores) != measure(y, old_scores)
            or not np.allclose(mel, fresh_mel, atol=1e-3, rtol=1e-3)
        ):
            raise ResearchError(f"Replay kontrol {domain} seed {seed} berbeda.")
        np.savez_compressed(
            output / f"control-s{seed}-{domain}.npz",
            indices=indices,
            scores=scores,
            fresh_scores=fresh_scores,
            y=y,
        )
        results.append(
            {
                "seed": seed,
                "domain": domain,
                "n": len(y),
                "correct": int((scores.argmax(1) == y).sum()),
                "accuracy": float((scores.argmax(1) == y).mean()),
                "cached_fresh_mel_max_abs_difference": float(np.abs(mel - fresh_mel).max()),
                "old_cpu_logit_max_abs_difference": float(np.abs(old_scores - scores).max()),
                "predictions_match": True,
            }
        )
    return results


def run(output):
    import torch

    from .resnet_fewshot import load_data

    torch.set_num_threads(4)
    sig = signature()
    config = {
        "input_sha256": sig,
        "seeds": list(SEEDS),
        "primary": "peak",
        "sensitivity": "tiled",
        "device": "cpu",
        "gpu_used": False,
        "threads": 4,
        "training_updates": 0,
        "target_fit_labels": 0,
        "target_normalization_fit": False,
        "class_order": KEYS,
    }
    if (output / "lock.json").exists():
        if read_json(output / "lock.json")["config"] != config:
            raise ResearchError("Lock berbeda; jangan melanjutkan/mengganti eksperimen.")
        if (output / "results.json").exists():
            result = read_json(output / "results.json")
            for name, digest in result["artifact_sha256"].items():
                if file_hash(output / name) != digest:
                    raise ResearchError(f"Artefak run berubah: {name}")
            return result
    else:
        if output.exists():
            raise ResearchError("Direktori keluaran sudah ada tanpa lock.")
        output.mkdir(parents=True)
        write_json(output / "lock.json", {"created_at": now(), "config": config})
        (output / "protocol.md").write_bytes(PROTOCOL.read_bytes())
    started = time.perf_counter()
    records, arrays = extract_inputs()
    write_json(output / "records.json", {"created_at": now(), "records": records})
    for policy in POLICIES:
        arrays[f"{policy}_mel"] = mel_cpu(arrays[f"{policy}_wave"])
    np.savez_compressed(output / "inputs.npz", **arrays)
    prepare_seconds = time.perf_counter() - started
    print(f"Prepared {len(records)} clips / {len(arrays['tiled_wave'])} tiled windows", flush=True)
    source_data, source_mel = load_data(
        ROOT / "data/external/harrison-fewshot", ROOT / "data/external/harrison-fewshot-features"
    )
    scores, source_controls, state_checks, timings = {}, [], [], []
    for seed in SEEDS:
        started = time.perf_counter()
        folder = SOURCE / f"sources/s{seed}"
        weights = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        model = initialize_model(seed, weights).requires_grad_(False).eval()
        before = state_digest(model.state_dict())
        with np.load(folder / "normalization.npz", allow_pickle=False) as stats:
            mean, std = (torch.from_numpy(stats[k]) for k in ("mean", "std"))
        source_controls.extend(controls(seed, model, mean, std, source_data, source_mel, output))
        for policy in POLICIES:
            values = _score(model, arrays[f"{policy}_mel"], mean, std, "cpu")
            if not np.isfinite(values).all():
                raise ResearchError("Logit non-finite.")
            scores[f"s{seed}_{policy}_window"] = values
            scores[f"s{seed}_{policy}_clip"] = (
                values
                if policy == "peak"
                else grouped_mean(values, arrays["tiled_owner"], len(records))
            )
        after = state_digest(model.state_dict())
        if before != after or before != state_digest(weights):
            raise ResearchError("Bobot/buffer berubah saat inferensi.")
        state_checks.append({"seed": seed, "before": before, "after": after})
        elapsed = time.perf_counter() - started
        timings.append({"seed": seed, "inference_and_controls_seconds": elapsed})
        print(f"Completed frozen seed {seed} in {elapsed:.1f}s; state unchanged", flush=True)
    np.savez_compressed(output / "scores.npz", **scores)
    # First computation of MKA metrics occurs only after all raw scores are saved.
    result = {
        "created_at": now(),
        "status": "complete",
        "config": config,
        "environment": environment(),
        "clips": len(records),
        "parent_groups": 216,
        "tiled_windows": len(arrays["tiled_wave"]),
        "peak_clamped": sum(r["peak_clamped"] for r in records),
        "prepare_seconds": prepare_seconds,
        "timings": timings,
        "source_controls": source_controls,
        "state_checks": state_checks,
        **measurements(records, scores),
    }
    if signature() != sig:
        raise ResearchError("Input/kode berubah selama evaluasi.")
    result["artifact_sha256"] = {
        p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()
    }
    write_json(output / "results.json", result)
    print("Completed zero-shot evaluation: no target fitting or GPU used", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/mka-zeroshot")
    args = parser.parse_args()
    try:
        run(args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
