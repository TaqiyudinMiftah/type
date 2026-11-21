"""Exploratory, explicitly budgeted channel adaptation on external recordings.

The experiment is specified in research/FEWSHOT_PROTOCOL.md. No target
validation set or target-driven checkpoint selection is used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

import joblib
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import normalize

from .common import (
    ResearchError,
    environment,
    file_hash,
    fresh_directory,
    now,
    read_json,
    write_json,
)
from .devices import configure_rx6000, select_device
from .harrison import DOMAINS, KEYS, SOURCE_COMMIT, _reference_model, augment, measure, spec_mask
from .harrison import split_indices as source_split

RATE = 16_000
SAMPLES = 6400
PRE_SAMPLES = 800
KS = (1, 2, 3, 5, 10)
SUPPORT_SEEDS = (101, 102, 103)
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "research/FEWSHOT_PROTOCOL.md"
BEATS_SHA256 = "d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34"


def calibration_split(y: np.ndarray, order: np.ndarray, k: int, seed: int) -> dict:
    """Nested supports; every later event is test-only regardless of draw/K."""
    if k not in KS or y.shape != order.shape:
        raise ResearchError("K atau bentuk label/urutan tidak sesuai protokol.")
    rng = np.random.default_rng(seed)
    support = []
    for label in range(len(KEYS)):
        indices = np.flatnonzero(y == label)
        if len(indices) != 25 or set(order[indices].tolist()) != set(range(25)):
            raise ResearchError("Setiap tombol harus memiliki tepat event 0–24 yang unik.")
        pool = indices[np.argsort(order[indices])][:15]
        support.extend(rng.permutation(pool)[:k].tolist())
    support = np.asarray(support, dtype=np.int64)
    test = np.flatnonzero(order >= 15)
    return {"support": support, "test": test}


def recrop(mono: np.ndarray, original_rate: int, centers: list[int]) -> tuple[np.ndarray, dict]:
    divisor = math.gcd(original_rate, RATE)
    audio = resample_poly(mono, RATE // divisor, original_rate // divisor).astype(np.float32)
    centers16 = np.rint(np.asarray(centers) * RATE / original_rate).astype(np.int64)
    starts = centers16 - PRE_SAMPLES
    ends = starts + SAMPLES
    if np.any(starts < 0) or np.any(ends > len(audio)):
        raise ResearchError("Potongan 400 ms melewati rekaman asli; padding tidak diizinkan.")
    if np.any(ends[:-1] > starts[1:]):
        raise ResearchError("Potongan bertetangga saling tumpang tindih.")
    waves = np.stack([audio[a:b] for a, b in zip(starts, ends, strict=True)])
    if not np.isfinite(waves).all():
        raise ResearchError("Audio non-finite.")
    return waves, {
        "centers_16k": centers16.tolist(),
        "starts_16k": starts.tolist(),
        "ends_16k": ends.tolist(),
        "minimum_center_gap_ms": float(np.diff(centers16).min() / RATE * 1000),
        "original_frames": len(mono),
        "resampled_frames": len(audio),
        "rms_min": float(np.sqrt(np.mean(waves**2, axis=1)).min()),
        "rms_max": float(np.sqrt(np.mean(waves**2, axis=1)).max()),
    }


def prepare(raw: Path, original_manifest: Path, output: Path) -> dict:
    manifest = read_json(original_manifest)
    if manifest["commit"] != SOURCE_COMMIT or manifest["keys"] != KEYS:
        raise ResearchError("Manifest Harrison tidak sesuai revisi/kelas.")
    if output.exists():
        raise ResearchError(f"Keluaran sudah ada: {output}")
    arrays, audit = {}, []
    hashes = set()
    for domain in DOMAINS:
        waves = []
        for key in KEYS:
            entry = next(f for f in manifest["files"] if f["domain"] == domain and f["key"] == key)
            path = raw / entry["path"]
            if file_hash(path) != entry["sha256"]:
                raise ResearchError(f"WAV asli berubah: {path}")
            audio, rate = sf.read(path, dtype="float32", always_2d=True)
            if rate != entry["sample_rate"] or len(entry["centers_samples"]) != 25:
                raise ResearchError(f"Metadata event tidak sesuai: {path}")
            clips, detail = recrop(audio.mean(axis=1), rate, entry["centers_samples"])
            for clip in clips:
                digest = hashlib.sha256(clip.tobytes()).hexdigest()
                if digest in hashes:
                    raise ResearchError("Duplikasi waveform persis setelah resampling.")
                hashes.add(digest)
            waves.append(clips)
            audit.append({**entry, **detail})
        arrays[f"{domain}_wave"] = np.concatenate(waves)
        arrays[f"{domain}_y"] = np.repeat(np.arange(36), 25)
        arrays[f"{domain}_order"] = np.tile(np.arange(25), 36)
        arrays[f"{domain}_ids"] = np.asarray(
            [f"{domain}:{key}:{i}" for key in KEYS for i in range(25)]
        )
    fresh_directory(output)
    np.savez_compressed(output / "segments.npz", **arrays)
    result = {
        "created_at": now(),
        "source_commit": SOURCE_COMMIT,
        "original_manifest_sha256": file_hash(original_manifest),
        "segments_sha256": file_hash(output / "segments.npz"),
        "rate": RATE,
        "samples": SAMPLES,
        "pre_samples": PRE_SAMPLES,
        "resample": "scipy.signal.resample_poly; default Kaiser beta=5; full recording first",
        "unique_waveforms": len(hashes),
        "files": audit,
        "protocol_sha256": file_hash(PROTOCOL),
        "environment": environment(),
    }
    write_json(output / "manifest.json", result)
    print(f"Prepared {len(hashes)} real clips at {RATE} Hz, {SAMPLES / RATE:.3f}s", flush=True)
    return result


@lru_cache(maxsize=4)
def _mel_constants(device):
    import torch

    return (
        torch.hann_window(512, device=device),
        torch.as_tensor(librosa.filters.mel(sr=RATE, n_fft=512, n_mels=64), device=device),
    )


def frontend(wave):
    import torch
    from torch.nn.functional import interpolate

    window, basis = _mel_constants(wave.device)
    power = (
        torch.stft(
            wave,
            512,
            hop_length=160,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        .abs()
        .square()
    )
    mel = 10 * (basis @ power).clamp_min(1e-10).log10()
    mel = torch.maximum(mel, mel.amax(dim=(-2, -1), keepdim=True) - 80)
    return interpolate(mel[:, None], size=(64, 64), mode="bilinear", align_corners=False)[:, 0]


def _beats(checkpoint: Path, device: str):
    import torch

    if file_hash(checkpoint) != BEATS_SHA256:
        raise ResearchError("Checkpoint BEATs tidak cocok dengan digest mirror yang dipatok.")
    vendor = ROOT / "research/vendor/beats"
    provenance = read_json(vendor / "provenance.json")
    for name, entry in provenance["files"].items():
        if file_hash(vendor / name) != entry["sha256"]:
            raise ResearchError(f"Sumber BEATs berubah: {name}")
    sys.path.insert(0, str(vendor))
    upstream = importlib.import_module("BEATs")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = upstream.BEATsConfig(saved["cfg"])
    if cfg.finetuned_model:
        raise ResearchError("Gunakan BEATs pretrained encoder, bukan head AudioSet fine-tuned.")
    model = upstream.BEATs(cfg)
    model.load_state_dict(saved["model"], strict=True)
    model.requires_grad_(False)
    model.eval().to(device)
    return model, provenance


def cache_features(prepared: Path, checkpoint: Path, output: Path) -> dict:
    configure_rx6000()
    device, hardware = select_device("gpu")
    import torch

    torch.set_num_threads(4)
    if output.exists():
        raise ResearchError(f"Cache sudah ada: {output}")
    metadata = read_json(prepared / "manifest.json")
    if file_hash(prepared / "segments.npz") != metadata["segments_sha256"]:
        raise ResearchError("Hash potongan tidak sesuai manifest.")
    model, provenance = _beats(checkpoint, device)
    data = np.load(prepared / "segments.npz", allow_pickle=False)
    arrays, timings, shapes = {}, {}, {}
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for domain in DOMAINS:
            torch.cuda.synchronize()
            start = time.perf_counter()
            vectors, images = [], []
            for offset in range(0, 900, 16):
                wave = torch.from_numpy(data[f"{domain}_wave"][offset : offset + 16]).to(device)
                tokens, mask = model.extract_features(wave)
                if mask is not None or tokens.shape[-1] != 768 or not torch.isfinite(tokens).all():
                    raise ResearchError("BEATs menghasilkan fitur tidak sesuai/finite.")
                vectors.append(tokens.mean(dim=1).cpu().numpy())
                images.append(frontend(wave).cpu().numpy())
                if offset == 0:
                    shapes[domain] = {
                        "fbank": list(model.preprocess(wave[:1]).shape),
                        "tokens": list(tokens[:1].shape),
                    }
                if offset % 160 == 0:
                    print(f"BEATs {domain}: {offset + len(wave)}/900", flush=True)
            torch.cuda.synchronize()
            timings[domain] = time.perf_counter() - start
            arrays[f"{domain}_beats"] = np.concatenate(vectors).astype(np.float32)
            arrays[f"{domain}_mel"] = np.concatenate(images).astype(np.float32)
    fresh_directory(output)
    np.savez_compressed(output / "features.npz", **arrays)
    result = {
        "created_at": now(),
        "segments_sha256": metadata["segments_sha256"],
        "features_sha256": file_hash(output / "features.npz"),
        "checkpoint_sha256": BEATS_SHA256,
        "checkpoint_provenance": read_json(checkpoint.parent / "provenance.json"),
        "vendor": provenance,
        "parameters": sum(p.numel() for p in model.parameters()),
        "shapes": shapes,
        "extraction_seconds_including_mel": timings,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "hardware": hardware,
        "protocol_sha256": file_hash(PROTOCOL),
        "environment": environment(),
    }
    write_json(output / "manifest.json", result)
    print(f"BEATs cache complete: {result['parameters']} parameters, {timings}", flush=True)
    return result


def _normalization(mel, indices):
    values = mel[indices]
    return (
        values.mean(axis=(0, 2), keepdims=True),
        np.maximum(values.std(axis=(0, 2), keepdims=True), 1e-6),
    )


def _seed(seed=42):
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def _score(model, mel, mean, std, device):
    import torch

    model.eval()
    values = []
    with torch.inference_mode():
        for offset in range(0, len(mel), 32):
            batch = torch.as_tensor(mel[offset : offset + 32], device=device)
            values.append(model(((batch - mean) / std)[:, None]).cpu().numpy())
    return np.concatenate(values)


def _source_train(data, features, domain, output, device):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if (output / "result.json").exists():
        return read_json(output / "result.json")
    output.mkdir(parents=True, exist_ok=True)
    _seed()
    wave, y, mel = (data[f"{domain}_wave"], data[f"{domain}_y"], features[f"{domain}_mel"])
    splits = source_split(y, data[f"{domain}_order"], seed=42)
    train, val, test = (splits[name] for name in ["train", "validation", "test"])
    mean, std = _normalization(mel, train)
    mean_t, std_t = (torch.as_tensor(x, device=device) for x in [mean, std])
    model, reference = _reference_model()
    model.to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(wave[train]), torch.from_numpy(y[train])),
        batch_size=16,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1, end_factor=0, total_iters=600
    )
    write_json(
        output / "config.json",
        {
            "domain": domain,
            "epochs": 600,
            "seed": 42,
            "reference": reference,
            "parameters": sum(p.numel() for p in model.parameters()),
            "split_ids": {name: data[f"{domain}_ids"][ix].tolist() for name, ix in splits.items()},
            "protocol_sha256": file_hash(PROTOCOL),
        },
    )
    np.savez(output / "normalization.npz", mean=mean, std=std)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    best_accuracy, best_epoch, history = -1, 0, []
    for epoch in range(1, 601):
        model.train()
        loss_sum = 0
        for batch, labels in loader:
            image = spec_mask(frontend(augment(batch.to(device))))
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(
                model(((image - mean_t) / std_t)[:, None]), labels.to(device)
            )
            if not torch.isfinite(loss):
                raise ResearchError(f"Source {domain}: loss non-finite.")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
        row = {"epoch": epoch, "loss": loss_sum / len(train)}
        if epoch == 1 or epoch % 5 == 0:
            scores = _score(model, mel[val], mean_t, std_t, device)
            acc = float(np.mean(scores.argmax(axis=1) == y[val]))
            row["validation_accuracy"] = acc
            if acc > best_accuracy:
                best_accuracy, best_epoch = acc, epoch
                torch.save(
                    {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    output / "best.pt",
                )
        scheduler.step()
        history.append(row)
        if epoch == 1 or epoch % 10 == 0:
            write_json(output / "history.json", {"epochs": history})
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"source {domain}: epoch {epoch}/600 loss={row['loss']:.4f} "
                f"best_val={best_accuracy:.4f} elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - start
    model.load_state_dict(torch.load(output / "best.pt", map_location="cpu", weights_only=True))
    scores = _score(model, mel[test], mean_t, std_t, device)
    np.savez_compressed(output / "source_test_scores.npz", scores=scores, y=y[test], indices=test)
    result = {
        "domain": domain,
        "training_seconds": training_seconds,
        "best_epoch": best_epoch,
        "validation_accuracy": best_accuracy,
        "source_test": measure(y[test], scores),
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "target_labels_used": 0,
        "source_train_labels": len(train),
        "source_validation_labels": len(val),
    }
    write_json(output / "result.json", result)
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def _target_train(data, features, source, target, k, seed, method, output, source_dir, device):
    import torch

    if (output / "result.json").exists():
        return read_json(output / "result.json")
    output.mkdir(parents=True, exist_ok=True)
    _seed()
    wave, y, mel = (data[f"{target}_wave"], data[f"{target}_y"], features[f"{target}_mel"])
    split = calibration_split(y, data[f"{target}_order"], k, seed)
    support, test = split["support"], split["test"]
    mean, std = _normalization(mel, support)
    mean_t, std_t = (torch.as_tensor(x, device=device) for x in [mean, std])
    model, _ = _reference_model()
    if method == "coatnet_adapt":
        model.load_state_dict(
            torch.load(source_dir / "best.pt", map_location="cpu", weights_only=True)
        )
    model.to(device)
    lr = 1e-4 if method == "coatnet_adapt" else 5e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1, end_factor=0, total_iters=1000
    )
    support_wave = torch.as_tensor(wave[support], device=device)
    support_y = torch.as_tensor(y[support], device=device)
    config = {
        "method": method,
        "source": source,
        "target": target,
        "k": k,
        "support_seed": seed,
        "training_seed": 42,
        "lr": lr,
        "updates": 1000,
        "batch_size": 16,
        "target_labels_used": len(support),
        "target_validation_labels": 0,
        "support_ids": data[f"{target}_ids"][support].tolist(),
        "test_ids": data[f"{target}_ids"][test].tolist(),
        "normalization": "target support only; identical for scratch and adaptation",
        "protocol_sha256": file_hash(PROTOCOL),
    }
    write_json(output / "config.json", config)
    np.savez(output / "normalization.npz", mean=mean, std=std)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    history = []
    model.train()
    for step in range(1, 1001):
        ix = torch.randint(len(support), (16,), device=device)
        image = spec_mask(frontend(augment(support_wave[ix])))
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(
            model(((image - mean_t) / std_t)[:, None]), support_y[ix]
        )
        if not torch.isfinite(loss):
            raise ResearchError(f"{output.name}: loss non-finite.")
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 100 == 0:
            history.append({"step": step, "loss": float(loss.item())})
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    support_scores = _score(model, mel[support], mean_t, std_t, device)
    torch.cuda.synchronize()
    inference_start = time.perf_counter()
    scores = _score(model, mel[test], mean_t, std_t, device)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_start
    result = {
        "method": method,
        "source": source,
        "target": target,
        "k": k,
        "support_seed": seed,
        "target_labels_used": len(support),
        "training_seconds": elapsed,
        "inference_seconds": inference_seconds,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "support_accuracy": float(np.mean(support_scores.argmax(axis=1) == y[support])),
        "last_augmented_loss": float(loss.item()),
        "test": measure(y[test], scores),
    }
    torch.save(
        {key: v.detach().cpu().clone() for key, v in model.state_dict().items()}, output / "last.pt"
    )
    np.savez_compressed(
        output / "scores.npz", scores=scores, y=y[test], indices=test, support=support
    )
    write_json(output / "history.json", {"updates": history})
    write_json(output / "result.json", result)
    print(
        f"{method} {source}->{target} K={k} seed={seed}: "
        f"test={result['test']['accuracy']:.4f} support={result['support_accuracy']:.4f} "
        f"{elapsed:.1f}s",
        flush=True,
    )
    del model, optimizer, support_wave, support_y
    torch.cuda.empty_cache()
    return result


def _linear(data, features, source, target, k, seed, output):
    if (output / "result.json").exists():
        return read_json(output / "result.json")
    output.mkdir(parents=True, exist_ok=True)
    target_y = data[f"{target}_y"]
    test = np.flatnonzero(data[f"{target}_order"] >= 15)
    if k:
        fit_domain = target
        support = calibration_split(target_y, data[f"{target}_order"], k, seed)["support"]
    else:
        fit_domain = source
        support = source_split(data[f"{source}_y"], data[f"{source}_order"], 42)["train"]
    start = time.perf_counter()
    model = RidgeClassifier(alpha=1.0)
    model.fit(normalize(features[f"{fit_domain}_beats"][support]), data[f"{fit_domain}_y"][support])
    training_seconds = time.perf_counter() - start
    start = time.perf_counter()
    scores = model.decision_function(normalize(features[f"{target}_beats"][test]))
    inference_seconds = time.perf_counter() - start
    result = {
        "method": "beats_linear",
        "source": source,
        "target": target,
        "k": k,
        "support_seed": seed,
        "target_labels_used": len(support) if k else 0,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "test": measure(target_y[test], scores),
    }
    write_json(
        output / "config.json",
        {
            **{
                key: result[key]
                for key in ["method", "source", "target", "k", "support_seed", "target_labels_used"]
            },
            "alpha": 1.0,
            "normalization": "L2 per vector; no data-fitted scaler",
            "fit_domain": fit_domain,
            "target_validation_labels": 0,
            "support_ids": data[f"{fit_domain}_ids"][support].tolist(),
            "test_ids": data[f"{target}_ids"][test].tolist(),
            "protocol_sha256": file_hash(PROTOCOL),
        },
    )
    np.savez_compressed(
        output / "scores.npz", scores=scores, y=target_y[test], indices=test, support=support
    )
    joblib.dump(model, output / "head.joblib")
    write_json(output / "result.json", result)
    print(
        f"BEATs linear {source}->{target} K={k} seed={seed}: {result['test']['accuracy']:.4f}",
        flush=True,
    )
    return result


def _zero_shot(data, features, source, target, output, source_dir, device):
    import torch

    if (output / "result.json").exists():
        return read_json(output / "result.json")
    output.mkdir(parents=True, exist_ok=True)
    stats = np.load(source_dir / "normalization.npz")
    mean, std = (torch.as_tensor(stats[key], device=device) for key in ["mean", "std"])
    model, _ = _reference_model()
    model.load_state_dict(torch.load(source_dir / "best.pt", map_location="cpu", weights_only=True))
    model.to(device)
    test = np.flatnonzero(data[f"{target}_order"] >= 15)
    torch.cuda.synchronize()
    start = time.perf_counter()
    scores = _score(model, features[f"{target}_mel"][test], mean, std, device)
    torch.cuda.synchronize()
    result = {
        "method": "coatnet_adapt",
        "source": source,
        "target": target,
        "k": 0,
        "support_seed": None,
        "target_labels_used": 0,
        "training_seconds": 0,
        "inference_seconds": time.perf_counter() - start,
        "test": measure(data[f"{target}_y"][test], scores),
    }
    np.savez_compressed(
        output / "scores.npz", scores=scores, y=data[f"{target}_y"][test], indices=test
    )
    write_json(output / "result.json", result)
    del model
    torch.cuda.empty_cache()
    return result


def run(prepared: Path, cache: Path, output: Path) -> dict:
    configure_rx6000()
    device, hardware = select_device("gpu")
    import torch

    torch.set_num_threads(4)
    data_meta, cache_meta = (read_json(p / "manifest.json") for p in [prepared, cache])
    if file_hash(prepared / "segments.npz") != data_meta["segments_sha256"]:
        raise ResearchError("Potongan berubah setelah audit.")
    if (
        file_hash(cache / "features.npz") != cache_meta["features_sha256"]
        or cache_meta["segments_sha256"] != data_meta["segments_sha256"]
    ):
        raise ResearchError("Cache tidak cocok dengan potongan.")
    if any(meta["protocol_sha256"] != file_hash(PROTOCOL) for meta in [data_meta, cache_meta]):
        raise ResearchError("Protokol berubah setelah persiapan/cache.")
    signature = {
        "protocol_sha256": file_hash(PROTOCOL),
        "segments_sha256": data_meta["segments_sha256"],
        "features_sha256": cache_meta["features_sha256"],
        "implementation_sha256": file_hash(Path(__file__)),
        "ks": list(KS),
        "support_seeds": list(SUPPORT_SEEDS),
    }
    if output.exists():
        if read_json(output / "manifest.json")["signature"] != signature:
            raise ResearchError("Resume ditolak: konfigurasi/data/implementasi berubah.")
    else:
        fresh_directory(output)
        write_json(
            output / "manifest.json",
            {
                "created_at": now(),
                "signature": signature,
                "hardware": hardware,
                "environment": environment(),
                "cache": cache_meta,
                "status": "running",
            },
        )
        (output / "protocol.md").write_text(PROTOCOL.read_text())
    data = np.load(prepared / "segments.npz", allow_pickle=False)
    features = np.load(cache / "features.npz", allow_pickle=False)
    # Materialize once: repeated NPZ decompression would contaminate fit timing.
    data = {name: data[name] for name in data.files}
    features = {name: features[name] for name in features.files}
    source_results, conditions = {}, []
    for domain in DOMAINS:
        source_results[domain] = _source_train(
            data, features, domain, output / f"source-{domain}", device
        )
    for source, target in [("phone", "zoom"), ("zoom", "phone")]:
        source_dir = output / f"source-{source}"
        conditions.append(
            _zero_shot(
                data,
                features,
                source,
                target,
                output / f"{source}-to-{target}-coatnet_adapt-k0",
                source_dir,
                device,
            )
        )
        conditions.append(
            _linear(
                data,
                features,
                source,
                target,
                0,
                None,
                output / f"{source}-to-{target}-beats_linear-k0",
            )
        )
        for k in KS:
            for seed in SUPPORT_SEEDS:
                conditions.append(
                    _linear(
                        data,
                        features,
                        source,
                        target,
                        k,
                        seed,
                        output / f"{source}-to-{target}-beats_linear-k{k}-s{seed}",
                    )
                )
                for method in ["coatnet_scratch", "coatnet_adapt"]:
                    conditions.append(
                        _target_train(
                            data,
                            features,
                            source,
                            target,
                            k,
                            seed,
                            method,
                            output / f"{source}-to-{target}-{method}-k{k}-s{seed}",
                            source_dir,
                            device,
                        )
                    )
                write_json(
                    output / "results.json",
                    {
                        "status": "running",
                        "sources": source_results,
                        "conditions": conditions,
                        "signature": signature,
                    },
                )
    result = {
        "status": "complete",
        "sources": source_results,
        "conditions": conditions,
        "signature": signature,
        "finished_at": now(),
    }
    write_json(output / "results.json", result)
    manifest = read_json(output / "manifest.json")
    write_json(output / "manifest.json", {**manifest, "status": "complete", "finished_at": now()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--raw", type=Path, required=True)
    prep.add_argument("--original-manifest", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    cache = commands.add_parser("cache")
    cache.add_argument("--prepared", type=Path, required=True)
    cache.add_argument("--checkpoint", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    experiment = commands.add_parser("run")
    experiment.add_argument("--prepared", type=Path, required=True)
    experiment.add_argument("--cache", type=Path, required=True)
    experiment.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(args.raw, args.original_manifest, args.output)
        elif args.command == "cache":
            cache_features(args.prepared, args.checkpoint, args.output)
        else:
            run(args.prepared, args.cache, args.output)
    except ResearchError as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
