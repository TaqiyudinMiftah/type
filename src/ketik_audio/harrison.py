"""Independent reproduction attempt on Harrison et al.'s original public recordings.

Isolation follows the author's MIT-licensed dataset README, with an audited
bounded-search extension. See research/licenses/harrison-MIT.txt for the notice
and research/LITERATURE.md for provenance and implementation differences.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import time
from pathlib import Path

import joblib
import librosa
import numpy as np
import soundfile as sf
from scipy.stats import norm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

KEYS = "1234567890qwertyuiopasdfghjklzxcvbnm"
SOURCE_COMMIT = "2ecdc09b6d1868aae50813f6a6c842bdd52dab06"
DOMAINS = {"phone": "MBPWavs", "zoom": "Zoom"}


def _positions(energy: np.ndarray, rate: int, threshold: float) -> np.ndarray:
    # Corresponds to README isolator(size=48, scan=24, before=2400, after=12000).
    timestamps = np.flatnonzero(energy > threshold) * 24 + 24
    positions = []
    cursor = 0
    while cursor < len(timestamps):
        timestamp = int(timestamps[cursor])
        positions.append(timestamp)
        cursor = int(np.searchsorted(timestamps, timestamp + 12_000 + rate * 0.1, side="right"))
    return np.asarray(positions, dtype=np.int64)


def isolate(signal: np.ndarray, rate: int, trim_seconds: int = 0) -> tuple[np.ndarray, dict]:
    trim = rate * trim_seconds
    working = signal[trim:]
    energy = np.abs(librosa.stft(working, n_fft=48, hop_length=24).sum(axis=0)).astype(float)
    threshold, step = 0.06, 0.005
    positions = np.empty(0, dtype=np.int64)
    strategy = "author-adaptive"
    initial_count = len(_positions(energy, rate, threshold))
    for iteration in range(2000):
        attempts = iteration + 1
        positions = _positions(energy, rate, threshold)
        if len(positions) == 25:
            break
        threshold += -step if len(positions) < 25 else step
        step *= 0.99
        if threshold <= 0:
            break
    else:
        attempts = 2000
    if len(positions) != 25:
        # The author's shrinking steps cannot leave [0, .56]. Extend the search
        # deterministically, without model outcomes or hand-selected timestamps.
        strategy = "extended-grid-4096-count-assisted"
        for candidate in np.linspace(1e-6, float(energy.max()), 4096):
            attempts += 1
            positions = _positions(energy, rate, float(candidate))
            if len(positions) == 25:
                threshold = float(candidate)
                break
        else:
            raise ResearchError("Tidak ditemukan ambang untuk 25 segmen; perlu audit manual.")
    if np.any(positions < 2400) or np.any(positions + 12_000 > len(working)):
        raise ResearchError("Segmen dari isolator penulis melewati batas rekaman.")
    clips = np.stack([working[p - 2400 : p + 12_000] for p in positions]).astype(np.float32)
    return clips, {
        "threshold": threshold,
        "iterations": attempts,
        "strategy": strategy,
        "initial_count": initial_count,
        "centers_samples": (positions + trim).tolist(),
        "trim_seconds": trim_seconds,
    }


def prepare(raw: Path, output: Path) -> dict:
    if output.exists():
        raise ResearchError(f"Keluaran sudah ada: {output}")
    revision = subprocess.run(
        ["git", "-C", str(raw), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if revision.returncode or revision.stdout.strip() != SOURCE_COMMIT:
        raise ResearchError(f"Dataset harus checkout commit {SOURCE_COMMIT}; lihat README.")
    provenance = []
    arrays = {}
    for domain, folder in DOMAINS.items():
        sources = {p.stem.lower(): p for p in (raw / folder).glob("*.wav")}
        if set(sources) != set(KEYS):
            raise ResearchError(f"{folder} harus memuat tepat 36 file a–z dan 0–9.")
        waves, targets, positions, source_ids, rates = [], [], [], [], set()
        for label, key in enumerate(KEYS):
            path = sources[key]
            audio, rate = sf.read(path, dtype="float32", always_2d=True)
            if not np.isfinite(audio).all():
                raise ResearchError(f"Nilai audio tidak finite: {path}")
            mono = audio.mean(axis=1)
            try:
                clips, audit = isolate(mono, rate, trim_seconds=1 if domain == "zoom" else 0)
            except ResearchError as exc:
                raise ResearchError(f"{domain}/{key}: {exc}") from exc
            waves.append(clips)
            targets.extend([label] * len(clips))
            positions.extend(range(len(clips)))
            source_ids.extend(f"{domain}:{key}:{i}" for i in range(len(clips)))
            rates.add(rate)
            provenance.append(
                {
                    "domain": domain,
                    "key": key,
                    "path": f"{folder}/{path.name}",
                    "sha256": file_hash(path),
                    "sample_rate": rate,
                    "channels": audio.shape[1],
                    "frames": len(audio),
                    "segments": len(clips),
                    **audit,
                }
            )
            print(
                f"{domain}/{key}: {len(clips)} segmen, threshold={audit['threshold']:.6f}",
                flush=True,
            )
        if len(rates) != 1:
            raise ResearchError(f"Sample rate dalam {domain} tidak seragam.")
        arrays[f"{domain}_wave"] = np.concatenate(waves)
        arrays[f"{domain}_y"] = np.asarray(targets, dtype=np.int64)
        arrays[f"{domain}_order"] = np.asarray(positions, dtype=np.int16)
        arrays[f"{domain}_ids"] = np.asarray(source_ids)
        arrays[f"{domain}_rate"] = np.asarray(rates.pop())
    fresh_directory(output)
    np.savez_compressed(output / "segments.npz", **arrays)
    manifest = {
        "created_at": now(),
        "source": "https://github.com/JBFH-Dev/Keystroke-Datasets",
        "commit": SOURCE_COMMIT,
        "license": "MIT",
        "keys": KEYS,
        "synthetic": False,
        "segment_samples": 14400,
        "isolation": "Author README FFT threshold; bounded search with audited grid fallback",
        "files": provenance,
        "segments_sha256": file_hash(output / "segments.npz"),
        "environment": environment(),
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def split_indices(y: np.ndarray, order: np.ndarray, seed: int, temporal: bool = False) -> dict:
    if temporal:
        splits = {
            "train": np.flatnonzero(order < 18),
            "validation": np.flatnonzero((order >= 18) & (order < 20)),
            "test": np.flatnonzero(order >= 20),
        }
    else:
        train_val, test = train_test_split(
            np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed
        )
        train, val = train_test_split(
            train_val, test_size=0.1, stratify=y[train_val], random_state=seed
        )
        splits = {"train": train, "validation": val, "test": test}
    combined = np.concatenate(list(splits.values()))
    if len(set(combined)) != len(y) or len(combined) != len(y):
        raise ResearchError("Split tidak membentuk partisi contoh asli.")
    return splits


def measure(y: np.ndarray, scores: np.ndarray) -> dict:
    pred = scores.argmax(axis=1)
    accuracy = float(accuracy_score(y, pred))
    n = len(y)
    z = float(norm.ppf(0.975))
    denominator = 1 + z * z / n
    center = (accuracy + z * z / (2 * n)) / denominator
    radius = z * np.sqrt(accuracy * (1 - accuracy) / n + z * z / (4 * n * n)) / denominator
    return {
        "n": n,
        "correct": int((pred == y).sum()),
        "accuracy": accuracy,
        "top5": float(top_k_accuracy_score(y, scores, k=5, labels=np.arange(36))),
        "macro_f1": float(
            f1_score(y, pred, labels=np.arange(36), average="macro", zero_division=0)
        ),
        "wilson95": [float(center - radius), float(center + radius)],
        "confusion_matrix": confusion_matrix(y, pred, labels=np.arange(36)).tolist(),
    }


def frontend(wave, rate: int):
    import torch
    from torch import nn

    window = torch.hann_window(1024, device=wave.device)
    power = (
        torch.stft(
            wave,
            1024,
            hop_length=225,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        .abs()
        .square()
    )
    basis = torch.as_tensor(librosa.filters.mel(sr=rate, n_fft=1024, n_mels=64), device=wave.device)
    mel = 10 * (basis @ power).clamp_min(1e-10).log10()
    mel = torch.maximum(mel, mel.amax(dim=(-2, -1), keepdim=True) - 80)
    return nn.functional.interpolate(
        mel[:, None], size=(64, 64), mode="bilinear", align_corners=False
    )[:, 0]


def cached_features(waves: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    import torch

    tensors = []
    with torch.inference_mode():
        for start in range(0, len(waves), 32):
            tensors.append(frontend(torch.from_numpy(waves[start : start + 32]), rate).numpy())
    mel = np.concatenate(tensors)
    vectors = []
    for image in mel:
        mfcc = librosa.feature.mfcc(S=image, n_mfcc=20)
        components = np.concatenate(
            [mfcc, librosa.feature.delta(mfcc), librosa.feature.delta(mfcc, order=2)]
        )
        vectors.append(np.concatenate([components.mean(axis=1), components.std(axis=1)]))
    return mel.astype(np.float32), np.asarray(vectors, dtype=np.float32)


def svm_experiment(x, y, splits, out: Path, target_x, target_y) -> dict:
    train, val, test = (splits[key] for key in ["train", "validation", "test"])
    start = time.perf_counter()
    best, score, chosen = None, -1.0, None
    for c in [1, 10, 100]:
        for gamma in ["scale", 0.01]:
            model = make_pipeline(StandardScaler(), SVC(C=c, gamma=gamma, break_ties=True))
            model.fit(x[train], y[train])
            candidate = f1_score(y[val], model.predict(x[val]), average="macro")
            if candidate > score:
                best, score, chosen = model, candidate, {"C": c, "gamma": gamma}
    result = {
        "kind": "mfcc-svm-control",
        "parameters": chosen,
        "validation_macro_f1": float(score),
        "training_seconds": time.perf_counter() - start,
        "test": measure(y[test], best.decision_function(x[test])),
        "cross_channel": measure(target_y, best.decision_function(target_x)),
    }
    out.mkdir()
    joblib.dump(best, out / "model.joblib")
    write_json(out / "result.json", result)
    return result


def _reference_model():
    import torch

    vendor = Path(__file__).resolve().parents[2] / "research/vendor/external_attention"
    if not (vendor / "provenance.json").exists():
        raise ResearchError("Jalankan uv run python research/fetch_reference.py terlebih dahulu.")
    provenance = read_json(vendor / "provenance.json")
    for filename, digest in provenance["files"].items():
        if file_hash(vendor / filename) != digest:
            raise ResearchError(f"Sumber backbone berubah: {filename}")
    sys.path.insert(0, str(vendor))
    backbone = importlib.import_module("model.attention.CoAtNet").CoAtNet(1, 64)
    return torch.nn.Sequential(
        backbone, torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(768, 36)
    ), provenance


def augment(wave):
    import torch

    batch, length = wave.shape
    shifts = torch.randint(
        -int(length * 0.4), int(length * 0.4) + 1, (batch, 1), device=wave.device
    )
    return wave.gather(1, (torch.arange(length, device=wave.device)[None] - shifts) % length)


def spec_mask(mel):
    import torch

    # Two masks per axis; same sample-specific mean fill as the paper description.
    fill = mel.mean(dim=(1, 2), keepdim=True)
    grid = torch.arange(64, device=mel.device)[None]
    for axis in [1, 2]:
        mask = torch.zeros((len(mel), 64), dtype=torch.bool, device=mel.device)
        for _ in range(2):
            width = torch.randint(0, 7, (len(mel), 1), device=mel.device)
            start = (torch.rand((len(mel), 1), device=mel.device) * (65 - width)).long()
            mask |= (grid >= start) & (grid < start + width)
        mel = torch.where(mask[:, :, None] if axis == 1 else mask[:, None, :], fill, mel)
    return mel


def neural_experiment(
    waves,
    mel,
    y,
    rate: int,
    splits: dict,
    target_mel,
    target_y,
    out: Path,
    device: str,
    epochs: int,
    seed: int,
) -> dict:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    train, val, test = (splits[key] for key in ["train", "validation", "test"])
    mean = mel[train].mean(axis=(0, 2), keepdims=True)
    std = np.maximum(mel[train].std(axis=(0, 2), keepdims=True), 1e-6)
    mean_t, std_t = (torch.from_numpy(value).to(device) for value in [mean, std])
    model, reference = _reference_model()
    model.to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(waves[train]), torch.from_numpy(y[train]).long()),
        batch_size=16,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs
    )
    validation_x = torch.from_numpy(mel[val]).to(device)

    def scores(images):
        model.eval()
        values = []
        with torch.inference_mode():
            for start in range(0, len(images), 32):
                batch = images[start : start + 32].to(device)
                values.append(model(((batch - mean_t) / std_t)[:, None]).cpu().numpy())
        return np.concatenate(values)

    out.mkdir()
    start = time.perf_counter()
    best, best_accuracy, best_epoch = None, -1.0, 0
    history = []
    config = {
        "kind": "coatnet-reference-reproduction-attempt",
        "epochs": epochs,
        "seed": seed,
        "batch_size": 16,
        "lr": 0.0005,
        "hop_length": 225,
        "channels": 1,
        "mel_shape": [64, 64],
        "resize": "bilinear",
        "normalization": "train-only mean/std per mel band",
        "schedule": "linear per epoch to zero",
        "validation_every": 5,
        "reference": reference,
        "parameters": sum(p.numel() for p in model.parameters()),
        "environment": environment(),
    }
    write_json(out / "config.json", config)
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for wave, labels in loader:
            features = spec_mask(frontend(augment(wave.to(device)), rate))
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(
                model(((features - mean_t) / std_t)[:, None]), labels.to(device)
            )
            if not torch.isfinite(loss):
                raise ResearchError(f"Loss bukan finite pada epoch {epoch}.")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(wave)
        row = {"epoch": epoch, "loss": loss_sum / len(train), "lr": optimizer.param_groups[0]["lr"]}
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            validation_scores = scores(validation_x)
            accuracy = float(accuracy_score(y[val], validation_scores.argmax(axis=1)))
            row["validation_accuracy"] = accuracy
            if accuracy > best_accuracy:
                best_accuracy, best_epoch = accuracy, epoch
                best = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                torch.save(best, out / "best.pt")
        scheduler.step()
        history.append(row)
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            write_json(out / "history.json", {"epochs": history})
        if epoch % 25 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"CoAtNet {out.name}: epoch {epoch}/{epochs}, loss={row['loss']:.4f}, "
                f"best_val={best_accuracy:.4f}, elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    model.load_state_dict(best)
    result = {
        "kind": config["kind"],
        "training_seconds": elapsed,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "test": measure(y[test], scores(torch.from_numpy(mel[test]))),
        "cross_channel": measure(target_y, scores(torch.from_numpy(target_mel))),
    }
    np.savez(out / "normalization.npz", mean=mean, std=std)
    write_json(out / "result.json", result)
    return result


def run(
    prepared: Path,
    output: Path,
    requested: str = "gpu",
    epochs: int = 1100,
    seed: int = 42,
    skip_neural: bool = False,
) -> dict:
    if epochs < 1:
        raise ResearchError("Jumlah epoch harus positif.")
    if requested != "cpu":
        configure_rx6000()
    device, device_info = select_device(requested)
    import torch

    torch.set_num_threads(4)
    manifest = read_json(prepared / "manifest.json")
    if file_hash(prepared / "segments.npz") != manifest["segments_sha256"]:
        raise ResearchError("Checksum segmen berubah.")
    with np.load(prepared / "segments.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    fresh_directory(output)
    config = {
        "created_at": now(),
        "device": device_info,
        "seed": seed,
        "epochs": epochs,
        "source_manifest": manifest,
        "environment": environment(),
        "neural_executed": not skip_neural,
    }
    write_json(output / "experiment.json", config)
    features = {}
    for domain in DOMAINS:
        features[domain] = cached_features(arrays[f"{domain}_wave"], int(arrays[f"{domain}_rate"]))
    results = []
    for domain in DOMAINS:
        target = "zoom" if domain == "phone" else "phone"
        mel, vectors = features[domain]
        target_mel, target_vectors = features[target]
        y, target_y = arrays[f"{domain}_y"], arrays[f"{target}_y"]
        for temporal in [False, True]:
            for baseline_seed in [42] if temporal else [42, 43, 44]:
                splits = split_indices(y, arrays[f"{domain}_order"], baseline_seed, temporal)
                tag = f"svm-{domain}-{'temporal' if temporal else 'random'}-{baseline_seed}"
                result = svm_experiment(vectors, y, splits, output / tag, target_vectors, target_y)
                result.update(
                    domain=domain,
                    split="temporal" if temporal else "random",
                    seed=baseline_seed,
                    directory=tag,
                )
                write_json(
                    output / tag / "split.json",
                    {
                        key: arrays[f"{domain}_ids"][indices].tolist()
                        for key, indices in splits.items()
                    },
                )
                results.append(result)
                print(
                    f"{tag}: test={result['test']['accuracy']:.4f}; "
                    f"cross={result['cross_channel']['accuracy']:.4f}",
                    flush=True,
                )
                write_json(output / "results.json", {"status": "running", "results": results})
        if not skip_neural:
            splits = split_indices(y, arrays[f"{domain}_order"], seed)
            tag = f"coatnet-{domain}-random-{seed}"
            result = neural_experiment(
                arrays[f"{domain}_wave"],
                mel,
                y,
                int(arrays[f"{domain}_rate"]),
                splits,
                target_mel,
                target_y,
                output / tag,
                device,
                epochs,
                seed,
            )
            result.update(domain=domain, split="random", seed=seed, directory=tag)
            write_json(
                output / tag / "split.json",
                {key: arrays[f"{domain}_ids"][indices].tolist() for key, indices in splits.items()},
            )
            results.append(result)
            write_json(output / "results.json", {"status": "running", "results": results})
    final = {"status": "complete", "results": results}
    write_json(output / "results.json", final)
    return final


def main():
    parser = argparse.ArgumentParser(
        description="Reproduksi independen Harrison 2023, dataset asli."
    )
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--raw", type=Path, default=Path("data/external/harrison-2023"))
    prep.add_argument("--output", type=Path, required=True)
    train = subs.add_parser("run")
    train.add_argument("--prepared", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", choices=["gpu", "cpu", "auto"], default="gpu")
    train.add_argument("--epochs", type=int, default=1100)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--skip-neural", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(args.raw, args.output)
        else:
            run(args.prepared, args.output, args.device, args.epochs, args.seed, args.skip_neural)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")


if __name__ == "__main__":
    main()
