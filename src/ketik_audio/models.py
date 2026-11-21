"""MFCC–SVM and small, portable CNN experiment artifacts."""

from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .common import (
    LABELS,
    SCHEMA_VERSION,
    FeatureConfig,
    ResearchError,
    environment,
    file_hash,
    fresh_directory,
    now,
    read_json,
    write_json,
)
from .dataset import load_prepared
from .devices import select_device
from .metrics import classify


def small_cnn():
    from torch import nn

    layers = []
    previous = 1
    for channels in [16, 32, 64]:
        layers.extend([nn.Conv2d(previous, channels, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)])
        previous = channels
    return nn.Sequential(
        *layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, len(LABELS))
    )


def _metadata(kind: str, manifest: dict, device: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "labels": list(LABELS),
        "created_at": now(),
        "synthetic": manifest["synthetic"],
        "feature_config": manifest["feature_config"],
        "source_manifest": manifest,
        "device": device,
        "environment": environment(),
    }


def train_svm(dataset: Path, output: Path) -> dict:
    arrays, manifest = load_prepared(dataset)
    if output.exists():
        raise ResearchError(f"Tujuan sudah ada: {output}.")
    train, val = arrays["sessions"] <= 4, arrays["sessions"] == 5
    start = time.perf_counter()
    best, best_score = None, -1.0
    candidates = []
    for c in [1, 10, 100]:
        for gamma in ["scale", 0.01]:
            pipeline = make_pipeline(
                StandardScaler(),
                SVC(C=c, gamma=gamma, kernel="rbf", break_ties=True, random_state=42),
            )
            pipeline.fit(arrays["mfcc"][train], arrays["y"][train])
            measured = classify(arrays["y"][val], pipeline.predict(arrays["mfcc"][val]))
            candidates.append({"C": c, "gamma": gamma, "macro_f1": measured["macro_f1"]})
            print(f"SVM C={c}, gamma={gamma}: validasi F1={measured['macro_f1']:.4f}", flush=True)
            if measured["macro_f1"] > best_score:
                best, best_score = pipeline, measured["macro_f1"]
    elapsed = time.perf_counter() - start
    fresh_directory(output)
    joblib.dump(best, output / "svm.joblib")
    metadata = _metadata("svm", manifest, {"requested": "cpu", "selected": "cpu"})
    metadata.update(
        {
            "artifact": "svm.joblib",
            "artifact_sha256": file_hash(output / "svm.joblib"),
            "score_kind": "softmax_of_svm_decision_not_calibrated",
            "selection": candidates,
            "training_seconds": elapsed,
            "selected_parameters": {
                "C": best.named_steps["svc"].C,
                "gamma": best.named_steps["svc"].gamma,
            },
        }
    )
    write_json(output / "model.json", metadata)
    result = {
        "kind": "svm",
        "validation": classify(arrays["y"][val], best.predict(arrays["mfcc"][val])),
        "training_seconds": elapsed,
    }
    write_json(output / "training.json", result)
    return result


def _seed_everything(seed: int, device: str):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)


def _cnn_scores(model, x, device: str, batch_size: int = 32) -> np.ndarray:
    import torch

    result = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            result.append(model(x[start : start + batch_size].to(device)).cpu().numpy())
    return np.concatenate(result) if result else np.empty((0, len(LABELS)), dtype=np.float32)


def train_cnn(
    dataset: Path,
    output: Path,
    requested: str = "auto",
    epochs: int = 50,
    patience: int = 8,
    batch_size: int = 32,
    seeds: tuple[int, ...] = (42, 43, 44),
):
    device, device_info = select_device(requested)
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if min(epochs, patience, batch_size) < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ResearchError("Epoch, patience, batch size harus positif; seed harus unik.")
    if any(seed < 0 or seed >= 2**32 for seed in seeds):
        raise ResearchError("Seed harus berada pada rentang 0 hingga 2**32-1.")
    arrays, manifest = load_prepared(dataset)
    if device_info.get("fallback_reason"):
        print(f"Menggunakan CPU: {device_info['fallback_reason']}", flush=True)
    fresh_directory(output)
    write_json(output / "training.json", {"status": "running", "device": device_info})
    train, val = arrays["sessions"] <= 4, arrays["sessions"] == 5
    mean = arrays["mel"][train].mean(axis=(0, 2), keepdims=True)
    std = np.maximum(arrays["mel"][train].std(axis=(0, 2), keepdims=True), 1e-6)
    normalized = ((arrays["mel"] - mean) / std)[:, None].astype(np.float32)
    x_train, x_val = torch.from_numpy(normalized[train]), torch.from_numpy(normalized[val])
    y_train = torch.from_numpy(arrays["y"][train]).long()
    seed_results = []
    selected = None
    total_start = time.perf_counter()
    for seed in seeds:
        _seed_everything(seed, device)
        model = small_cnn().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        best_score, stale, best_state, best_epoch = -1.0, 0, None, 0
        history = []
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum = 0.0
            for x_batch, y_batch in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x_batch.to(device)), y_batch.to(device))
                if not torch.isfinite(loss).item():
                    raise ResearchError("Loss tidak finite. Periksa dataset dan backend perangkat.")
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(x_batch)
            scores = _cnn_scores(model, x_val, device, batch_size)
            measured = classify(arrays["y"][val], scores.argmax(axis=1))
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": loss_sum / len(x_train),
                    "validation_accuracy": measured["accuracy"],
                    "validation_macro_f1": measured["macro_f1"],
                }
            )
            print(
                f"CNN seed={seed} epoch={epoch}/{epochs} "
                f"loss={history[-1]['train_loss']:.4f} "
                f"validasi F1={measured['macro_f1']:.4f}",
                flush=True,
            )
            if measured["macro_f1"] > best_score:
                best_score, stale, best_epoch = measured["macro_f1"], 0, epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
            else:
                stale += 1
            if stale >= patience:
                break
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        seed_path = output / f"seed-{seed}"
        seed_path.mkdir()
        torch.save(best_state, seed_path / "weights.pt")
        metadata = _metadata("cnn", manifest, device_info)
        metadata.update(
            {
                "seed": seed,
                "best_epoch": best_epoch,
                "max_epochs": epochs,
                "patience": patience,
                "batch_size": batch_size,
                "learning_rate": 0.001,
                "normalization_mean": mean.reshape(-1).tolist(),
                "normalization_std": std.reshape(-1).tolist(),
                "artifact": "weights.pt",
                "artifact_sha256": file_hash(seed_path / "weights.pt"),
                "score_kind": "softmax_not_calibrated",
                "training_seconds": elapsed,
            }
        )
        write_json(seed_path / "model.json", metadata)
        model.load_state_dict(best_state)
        validation = classify(
            arrays["y"][val], _cnn_scores(model, x_val, device, batch_size).argmax(axis=1)
        )
        result = {
            "seed": seed,
            "best_epoch": best_epoch,
            "training_seconds": elapsed,
            "validation": validation,
            "history": history,
        }
        write_json(seed_path / "training.json", result)
        seed_results.append(result)
        if selected is None or best_score > selected[0]:
            selected = (best_score, seed, copy.deepcopy(metadata))
    # Selection is based solely on session 5, never on the test recording.
    metadata = selected[2]
    metadata["artifact"] = f"seed-{selected[1]}/weights.pt"
    metadata["seed_models"] = [f"seed-{seed}" for seed in seeds]
    write_json(output / "model.json", metadata)
    result = {
        "status": "complete",
        "kind": "cnn",
        "selected_seed": selected[1],
        "device": device_info,
        "seeds": seed_results,
        "training_seconds": time.perf_counter() - total_start,
        "validation_macro_f1_mean": float(
            np.mean([r["validation"]["macro_f1"] for r in seed_results])
        ),
        "validation_macro_f1_std": float(
            np.std([r["validation"]["macro_f1"] for r in seed_results])
        ),
    }
    write_json(output / "training.json", result)
    return result


class Predictor:
    def __init__(self, path: Path, requested: str = "cpu"):
        self.path = path
        self.metadata = read_json(path / "model.json")
        if self.metadata.get("schema_version") != SCHEMA_VERSION or self.metadata.get(
            "labels"
        ) != list(LABELS):
            raise ResearchError("Versi model/pemetaan huruf tidak didukung.")
        self.config = FeatureConfig.from_dict(self.metadata["feature_config"])
        artifact = (path / self.metadata["artifact"]).resolve()
        if not artifact.is_relative_to(path.resolve()):
            raise ResearchError("Artefak model berada di luar direktori model.")
        if file_hash(artifact) != self.metadata["artifact_sha256"]:
            raise ResearchError("Artefak model berubah/rusak; checksum tidak cocok.")
        self.kind = self.metadata["kind"]
        if self.kind == "svm":
            if requested == "gpu":
                raise ResearchError("SVM berjalan pada CPU. Gunakan CNN untuk pengujian GPU.")
            self.device = "cpu"
            self.device_info = {"selected": "cpu", "requested": requested}
            self.model = joblib.load(artifact)
        elif self.kind == "cnn":
            self.device, self.device_info = select_device(requested)
            import torch

            torch.set_num_threads(4)
            self.model = small_cnn()
            self.model.load_state_dict(torch.load(artifact, map_location="cpu", weights_only=True))
            self.model.to(self.device).eval()
            self.mean = np.asarray(self.metadata["normalization_mean"], dtype=np.float32)[
                None, :, None
            ]
            self.std = np.asarray(self.metadata["normalization_std"], dtype=np.float32)[
                None, :, None
            ]
        else:
            raise ResearchError(f"Jenis model tidak dikenal: {self.kind}")

    def scores(self, mel: np.ndarray, mfcc: np.ndarray) -> np.ndarray:
        if not len(mel):
            return np.empty((0, len(LABELS)), dtype=np.float32)
        if self.kind == "svm":
            scores = self.model.decision_function(mfcc)
        else:
            import torch

            x = torch.from_numpy((((mel - self.mean) / self.std)[:, None]).astype(np.float32))
            scores = _cnn_scores(self.model, x, self.device)
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def synchronize(self):
        if self.device == "cuda":
            import torch

            torch.cuda.synchronize()
