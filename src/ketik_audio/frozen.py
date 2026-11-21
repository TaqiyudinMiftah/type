"""Linear probes with frozen CoAtNet representations and matched label budgets."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
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
from .fewshot import KS, ROOT, SUPPORT_SEEDS, _normalization, calibration_split
from .harrison import DOMAINS, _reference_model, measure, split_indices

PROTOCOL = ROOT / "research/FROZEN_HEAD_PROTOCOL.md"
METHODS = ("pretrained_source_norm", "random_source_norm", "pretrained_support_norm")


def frozen_encoder(model):
    """Freeze parameters AND stateful buffers; gradient mode alone is insufficient."""
    import torch

    model.requires_grad_(False).eval()
    return torch.nn.Sequential(*list(model.children())[:-1]).eval()


def extract(encoder, mel, mean, std, device):
    import torch

    if any(module.training for module in encoder.modules()):
        raise ResearchError("Encoder beku tidak boleh dalam mode train.")
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise ResearchError("Parameter encoder belum dibekukan.")
    mean_t, std_t = (torch.as_tensor(x, device=device) for x in [mean, std])
    values = []
    with torch.inference_mode():
        for offset in range(0, len(mel), 32):
            image = torch.as_tensor(mel[offset : offset + 32], device=device)
            vector = encoder(((image - mean_t) / std_t)[:, None])
            if vector.ndim != 2 or not torch.isfinite(vector).all():
                raise ResearchError("Fitur encoder bukan matriks finite.")
            values.append(vector.cpu().numpy())
    return np.concatenate(values)


def state_snapshot(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def assert_unchanged(model, before):
    import torch

    after = model.state_dict()
    if after.keys() != before.keys() or any(
        not torch.equal(before[k], after[k].cpu()) for k in before
    ):
        raise ResearchError("Bobot/buffer encoder berubah saat ekstraksi target.")


def fit_head(fit_x, fit_y, test_x):
    start = time.perf_counter()
    head = RidgeClassifier(alpha=1.0)
    head.fit(normalize(fit_x), fit_y)
    elapsed = time.perf_counter() - start
    start = time.perf_counter()
    scores = head.decision_function(normalize(test_x))
    return head, scores, elapsed, time.perf_counter() - start


def run(primary: Path, prepared: Path, cache: Path, output: Path):
    configure_rx6000()
    device, hardware = select_device("gpu")
    import torch

    torch.set_num_threads(4)
    previous = read_json(primary / "results.json")
    if previous["status"] != "complete":
        raise ResearchError("Pilot sumber harus selesai.")
    if file_hash(prepared / "segments.npz") != previous["signature"]["segments_sha256"]:
        raise ResearchError("Potongan berubah dari pilot.")
    if file_hash(cache / "features.npz") != previous["signature"]["features_sha256"]:
        raise ResearchError("Cache mel berubah dari pilot.")
    fresh_directory(output)
    signature = {
        "primary_results_sha256": file_hash(primary / "results.json"),
        "segments_sha256": previous["signature"]["segments_sha256"],
        "input_features_sha256": previous["signature"]["features_sha256"],
        "protocol_sha256": file_hash(PROTOCOL),
        "implementation_sha256": file_hash(Path(__file__)),
        "fewshot_implementation_sha256": file_hash(ROOT / "src/ketik_audio/fewshot.py"),
        "harrison_implementation_sha256": file_hash(ROOT / "src/ketik_audio/harrison.py"),
        "source_weights_sha256": {s: file_hash(primary / f"source-{s}/best.pt") for s in DOMAINS},
        "source_norm_sha256": {
            s: file_hash(primary / f"source-{s}/normalization.npz") for s in DOMAINS
        },
    }
    write_json(
        output / "manifest.json",
        {
            "status": "running",
            "created_at": now(),
            "signature": signature,
            "hardware": hardware,
            "environment": environment(),
        },
    )
    (output / "protocol.md").write_text(PROTOCOL.read_text())
    with np.load(prepared / "segments.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    with np.load(cache / "features.npz", allow_pickle=False) as archive:
        images = {s: archive[f"{s}_mel"] for s in DOMAINS}
    conditions, encoders, source_checks = [], [], []
    for source, target in [("phone", "zoom"), ("zoom", "phone")]:
        source_dir = primary / f"source-{source}"
        with np.load(source_dir / "normalization.npz", allow_pickle=False) as stats:
            mean, std = stats["mean"], stats["std"]
        train_split = split_indices(data[f"{source}_y"], data[f"{source}_order"], 42)
        train, source_test = train_split["train"], train_split["test"]
        test = np.flatnonzero(data[f"{target}_order"] >= 15)
        for kind in ["pretrained", "random"]:
            torch.manual_seed(42)
            model, reference = _reference_model()
            if kind == "pretrained":
                model.load_state_dict(
                    torch.load(source_dir / "best.pt", map_location="cpu", weights_only=True)
                )
            encoder = frozen_encoder(model).to(device)
            before = state_snapshot(encoder)
            enc_dir = output / "encoders" / f"{source}-{kind}"
            enc_dir.mkdir(parents=True)
            torch.save(state_snapshot(model), enc_dir / "model.pt")
            np.savez(enc_dir / "normalization.npz", mean=mean, std=std)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            vectors = {
                domain: extract(encoder, images[domain], mean, std, device) for domain in DOMAINS
            }
            torch.cuda.synchronize()
            extraction_seconds = time.perf_counter() - start
            assert_unchanged(encoder, before)
            np.savez_compressed(enc_dir / "features.npz", **vectors)
            encoder_record = {
                "source": source,
                "kind": kind,
                "seed": 42,
                "feature_dimension": vectors[source].shape[1],
                "extraction_seconds": extraction_seconds,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                "weights_sha256": file_hash(enc_dir / "model.pt"),
                "features_sha256": file_hash(enc_dir / "features.npz"),
                "normalization_sha256": file_hash(enc_dir / "normalization.npz"),
                "state_unchanged": True,
                "batchnorm_eval": True,
                "reference": reference,
            }
            encoders.append(encoder_record)
            write_json(enc_dir / "config.json", encoder_record)
            head, scores, _, _ = fit_head(
                vectors[source][train], data[f"{source}_y"][train], vectors[source][source_test]
            )
            source_checks.append(
                {
                    "source": source,
                    "kind": kind,
                    "test": measure(data[f"{source}_y"][source_test], scores),
                }
            )
            for k in [0, *KS]:
                for seed in [None] if k == 0 else SUPPORT_SEEDS:
                    support = (
                        train
                        if k == 0
                        else calibration_split(
                            data[f"{target}_y"], data[f"{target}_order"], k, seed
                        )["support"]
                    )
                    fit_domain = source if k == 0 else target
                    variants = [f"{kind}_source_norm"]
                    if kind == "pretrained" and k:
                        variants.append("pretrained_support_norm")
                    for method in variants:
                        suffix = f"-s{seed}" if k else ""
                        out = output / f"{source}-to-{target}-{method}-k{k}{suffix}"
                        out.mkdir()
                        condition_mean, condition_std = mean, std
                        condition_extract_seconds = 0.0
                        if method == "pretrained_support_norm":
                            condition_mean, condition_std = _normalization(images[target], support)
                            indices = np.concatenate([support, test])
                            torch.cuda.synchronize()
                            start = time.perf_counter()
                            transformed = extract(
                                encoder,
                                images[target][indices],
                                condition_mean,
                                condition_std,
                                device,
                            )
                            torch.cuda.synchronize()
                            condition_extract_seconds = time.perf_counter() - start
                            fit_x, test_x = transformed[: len(support)], transformed[len(support) :]
                            np.savez_compressed(out / "vectors.npz", fit=fit_x, test=test_x)
                        else:
                            fit_x, test_x = vectors[fit_domain][support], vectors[target][test]
                        head, scores, fit_seconds, inference_seconds = fit_head(
                            fit_x, data[f"{fit_domain}_y"][support], test_x
                        )
                        support_pred = head.predict(normalize(fit_x))
                        row = {
                            "source": source,
                            "target": target,
                            "method": method,
                            "k": k,
                            "support_seed": seed,
                            "target_labels_used": k * 36,
                            "training_seconds": fit_seconds,
                            "inference_seconds": inference_seconds,
                            "extra_extraction_seconds": condition_extract_seconds,
                            "support_accuracy": float(
                                np.mean(support_pred == data[f"{fit_domain}_y"][support])
                            ),
                            "test": measure(data[f"{target}_y"][test], scores),
                            "directory": out.name,
                        }
                        joblib.dump(head, out / "head.joblib")
                        np.savez(out / "normalization.npz", mean=condition_mean, std=condition_std)
                        np.savez_compressed(
                            out / "scores.npz",
                            scores=scores,
                            y=data[f"{target}_y"][test],
                            indices=test,
                            support=support,
                        )
                        write_json(
                            out / "config.json",
                            {
                                "method": method,
                                "encoder_directory": str(enc_dir.relative_to(output)),
                                "fit_domain": fit_domain,
                                "alpha": 1.0,
                                "target_labels_used": k * 36,
                                "target_validation_labels": 0,
                                "source": source,
                                "target": target,
                                "k": k,
                                "support_seed": seed,
                                "feature_normalization": "per-vector L2",
                                "support_ids": data[f"{fit_domain}_ids"][support].tolist(),
                                "test_ids": data[f"{target}_ids"][test].tolist(),
                                "protocol_sha256": signature["protocol_sha256"],
                            },
                        )
                        write_json(out / "result.json", row)
                        conditions.append(row)
                        print(
                            f"{source}->{target} {method} K={k} seed={seed}: "
                            f"{row['test']['accuracy']:.4f}",
                            flush=True,
                        )
            assert_unchanged(encoder, before)
            del encoder, model
            torch.cuda.empty_cache()
    result = {
        "status": "complete",
        "finished_at": now(),
        "signature": signature,
        "encoders": encoders,
        "source_checks": source_checks,
        "conditions": conditions,
    }
    write_json(output / "results.json", result)
    meta = read_json(output / "manifest.json")
    write_json(output / "manifest.json", {**meta, "status": "complete", "finished_at": now()})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.primary, args.prepared, args.cache, args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
