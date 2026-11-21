"""Paired label-budget study of source initialization versus scratch (Zoom→phone)."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

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
from .fewshot import KS, ROOT, _normalization, _score, calibration_split, frontend
from .harrison import augment, measure, spec_mask, split_indices

PROTOCOL = ROOT / "research/RESNET_FEWSHOT_PROTOCOL.md"
BLOCKS = ((42, 101), (43, 102), (44, 103))
UPDATES = 10_000
PROBE_STEPS = (1, 100, 1000, 3000, 6000, 10_000)
MODEL_SHA = "f7c341379c6b030254cf31a3a293edcf7c4f8a42673d272c910da91fd980e523"
PARENT_SHA = "4b54a8aa3230a651588362ed928b2113b1bf5a709793eeda18c0803d5d39bf3c"


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def state_digest(state):
    """Hash tensor values, names and shapes, independently of torch.save's container."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        descriptor = {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
        digest.update(json.dumps(descriptor, sort_keys=True).encode())
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def initialize_model(seed, source_state=None):
    from .cnn_repair import repaired_model

    model, _ = repaired_model(seed, "compact-resnet")
    if source_state is not None:
        model.load_state_dict(source_state, strict=True)
    return model


def seed_training(seed):
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def training_batch(wave, labels, mean, std):
    import torch

    indices = torch.randint(len(wave), (16,), device=wave.device)
    image = spec_mask(frontend(augment(wave[indices])))
    return indices, ((image - mean) / std)[:, None], labels[indices]


def atomic_torch_save(value, path):
    import torch

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def save_recovery(path, model, optimizer, scheduler, step, metadata):
    import torch

    atomic_torch_save(
        {
            "model": cpu_state(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "gpu_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "step": step,
            "metadata": metadata,
        },
        path,
    )


def load_recovery(path, model, optimizer, scheduler):
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    torch.set_rng_state(saved["cpu_rng"])
    if saved["gpu_rng"]:
        if not torch.cuda.is_available() or len(saved["gpu_rng"]) != torch.cuda.device_count():
            raise ResearchError("Perangkat GPU pemulihan berbeda.")
        torch.cuda.set_rng_state_all(saved["gpu_rng"])
    return saved["step"], saved["metadata"]


def signature(prepared, cache):
    parent = read_json(ROOT / "runs/cnn-repair-confirmation/results.json")
    paths = {
        "implementation_sha256": Path(__file__),
        "protocol_sha256": PROTOCOL,
        "model_implementation_sha256": ROOT / "src/ketik_audio/cnn_repair.py",
        "fewshot_implementation_sha256": ROOT / "src/ketik_audio/fewshot.py",
        "harrison_implementation_sha256": ROOT / "src/ketik_audio/harrison.py",
        "segments_sha256": prepared / "segments.npz",
        "features_sha256": cache / "features.npz",
        "parent_results_sha256": ROOT / "runs/cnn-repair-confirmation/results.json",
    }
    result = {name: file_hash(path) for name, path in paths.items()}
    if (
        result["model_implementation_sha256"] != MODEL_SHA
        or result["parent_results_sha256"] != PARENT_SHA
    ):
        raise ResearchError("Arsitektur/baseline stabil berubah.")
    for field in (
        "segments_sha256",
        "features_sha256",
        "fewshot_implementation_sha256",
        "harrison_implementation_sha256",
    ):
        if result[field] != parent["signature"][field]:
            raise ResearchError(f"Input/metode baseline berubah: {field}")
    return result


def check_code(signature_record):
    for path, field in (
        (Path(__file__), "implementation_sha256"),
        (PROTOCOL, "protocol_sha256"),
        (ROOT / "src/ketik_audio/cnn_repair.py", "model_implementation_sha256"),
    ):
        if file_hash(path) != signature_record[field]:
            raise ResearchError(f"Kode/protokol berubah selama eksperimen: {path}")


def load_data(prepared, cache):
    with np.load(prepared / "segments.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    with np.load(cache / "features.npz", allow_pickle=False) as archive:
        mel = {domain: archive[f"{domain}_mel"] for domain in ("phone", "zoom")}
    return data, mel


def fit(data, mel, domain, split, seed, support_seed, k, method, output, device, sig, source=None):
    """One shared training loop: only its initial state differs between target methods."""
    import torch

    stage = "source" if method == "source" else "target"
    support, evaluation = split["support"], split["test"]
    y, ids = data[f"{domain}_y"], data[f"{domain}_ids"]
    source_sha = file_hash(source / "best.pt") if source is not None else None
    config = {
        "signature": sig,
        "stage": stage,
        "domain": domain,
        "method": method,
        "training_seed": seed,
        "support_seed": support_seed,
        "k": k,
        "updates": UPDATES,
        "batch_size": 16,
        "lr": 5e-4,
        "scheduler": "linear_to_zero",
        "optimizer": "Adam",
        "optimizer_initial_state_entries": 0,
        "target_labels_used": len(support) if stage == "target" else 0,
        "target_validation_labels": 0,
        "source_checkpoint_sha256": source_sha,
        "source_directory": str(source.relative_to(output.parents[1]))
        if source is not None
        else None,
        "support_ids": ids[support].tolist(),
        "test_ids": ids[evaluation].tolist(),
        "normalization_fit_on": "support_only",
        "selection": "source_validation" if stage == "source" else "last_update",
    }
    if stage == "source":
        config["validation_ids"] = ids[split["validation"]].tolist()
        config["source_train_labels"], config["source_validation_labels"] = (
            len(support),
            len(split["validation"]),
        )
    if (output / "result.json").exists():
        saved_config = read_json(output / "config.json")
        if any(saved_config.get(key) != value for key, value in config.items()):
            raise ResearchError(f"Konfigurasi hasil selesai berbeda: {output}")
        result = read_json(output / "result.json")
        if result["weights_sha256"] != file_hash(output / result["checkpoint"]):
            raise ResearchError("Checkpoint hasil selesai berubah.")
        return result
    state = (
        torch.load(source / "best.pt", map_location="cpu", weights_only=True) if source else None
    )
    model = initialize_model(seed, state).to(device)
    config["initial_state_digest"] = state_digest(model.state_dict())
    config["parameters"] = sum(p.numel() for p in model.parameters())
    mean, std = _normalization(mel, support)
    mean_t, std_t = (torch.as_tensor(x, device=device) for x in (mean, std))
    wave = torch.as_tensor(data[f"{domain}_wave"][support], device=device)
    labels = torch.as_tensor(y[support], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1, end_factor=0, total_iters=UPDATES
    )
    if optimizer.state:
        raise ResearchError("Optimizer baru tidak kosong.")
    seed_training(seed)
    torch.cuda.reset_peak_memory_stats()
    initial_step = 0
    metadata = {
        "history": [],
        "batch_probes": [],
        "best_accuracy": -1.0,
        "best_step": 0,
        "best_state": None,
        "elapsed_seconds": 0.0,
        "peak_gpu_allocated_bytes": 0,
    }
    if output.exists():
        if read_json(output / "config.json") != config or not (output / "recovery.pt").exists():
            raise ResearchError(f"Fit parsial tanpa checkpoint/config yang cocok: {output}")
        initial_step, metadata = load_recovery(output / "recovery.pt", model, optimizer, scheduler)
        print(f"RESUME {stage} {method} seed={seed} K={k} update={initial_step}", flush=True)
    else:
        fresh_directory(output)
        write_json(output / "config.json", config)
        np.savez(output / "normalization.npz", mean=mean, std=std)
        save_recovery(output / "recovery.pt", model, optimizer, scheduler, 0, metadata)
    torch.cuda.synchronize()
    start = time.perf_counter()
    previous_elapsed = metadata["elapsed_seconds"]
    for step in range(initial_step + 1, UPDATES + 1):
        model.train()
        ix, image, batch_labels = training_batch(wave, labels, mean_t, std_t)
        if step in PROBE_STEPS:
            digest = hashlib.sha256(ix.cpu().numpy().tobytes())
            digest.update(image.detach().cpu().numpy().tobytes())
            metadata["batch_probes"].append({"step": step, "sha256": digest.hexdigest()})
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(image), batch_labels)
        if not torch.isfinite(loss):
            raise ResearchError(f"Loss non-finite: {stage}/{method}/{seed}/{k}/{step}")
        loss.backward()
        gradient = (
            float(model[0].s0[0].weight.grad.detach().double().norm().item())
            if step in PROBE_STEPS
            else None
        )
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 500 == 0 or gradient is not None:
            row = {"step": step, "loss": float(loss.item())}
            if gradient is not None:
                row["stem_gradient_l2"] = gradient
            if step in (1000, 3000, 6000, UPDATES):
                scores = _score(model, mel[support], mean_t, std_t, device)
                row["support_accuracy"] = float((scores.argmax(1) == y[support]).mean())
            if stage == "source" and step % 500 == 0:
                validation = split["validation"]
                scores = _score(model, mel[validation], mean_t, std_t, device)
                accuracy = float((scores.argmax(1) == y[validation]).mean())
                row["validation_accuracy"] = accuracy
                if accuracy > metadata["best_accuracy"]:
                    metadata.update(
                        best_accuracy=accuracy, best_step=step, best_state=cpu_state(model)
                    )
                    atomic_torch_save(metadata["best_state"], output / "best.pt")
            metadata["history"].append(row)
            write_json(output / "history.json", {"updates": metadata["history"]})
            if step == 1 or step % 1000 == 0:
                print(
                    f"{stage} {method} seed={seed} draw={support_seed} K={k} "
                    f"{step}/{UPDATES} loss={row['loss']:.4f} "
                    f"support={row.get('support_accuracy')} "
                    f"validation={row.get('validation_accuracy')}",
                    flush=True,
                )
        if step % 1000 == 0:
            torch.cuda.synchronize()
            metadata["elapsed_seconds"] = previous_elapsed + time.perf_counter() - start
            metadata["peak_gpu_allocated_bytes"] = max(
                metadata["peak_gpu_allocated_bytes"], torch.cuda.max_memory_allocated()
            )
            save_recovery(output / "recovery.pt", model, optimizer, scheduler, step, metadata)
    torch.cuda.synchronize()
    elapsed = previous_elapsed + time.perf_counter() - start
    atomic_torch_save(cpu_state(model), output / "last.pt")
    checkpoint = "best.pt" if stage == "source" else "last.pt"
    if stage == "source":
        if metadata["best_state"] is None:
            raise ResearchError("Tidak ada checkpoint sumber yang tervalidasi.")
        model.load_state_dict(metadata["best_state"], strict=True)
        atomic_torch_save(metadata["best_state"], output / "best.pt")
    support_scores = _score(model, mel[support], mean_t, std_t, device)
    scores = _score(model, mel[evaluation], mean_t, std_t, device)
    np.savez_compressed(
        output / "scores.npz",
        scores=scores,
        y=y[evaluation],
        indices=evaluation,
        support=support,
        support_scores=support_scores,
    )
    result = {
        "stage": stage,
        "domain": domain,
        "method": method,
        "training_seed": seed,
        "support_seed": support_seed,
        "k": k,
        "updates": UPDATES,
        "support_accuracy": float((support_scores.argmax(1) == y[support]).mean()),
        "support_max_logit_range": float(np.ptp(support_scores, axis=0).max()),
        "test": measure(y[evaluation], scores),
        "checkpoint": checkpoint,
        "weights_sha256": file_hash(output / checkpoint),
        "normalization_sha256": file_hash(output / "normalization.npz"),
        "initial_state_digest": config["initial_state_digest"],
        "source_checkpoint_sha256": source_sha,
        "batch_probes": metadata["batch_probes"],
        "training_seconds": elapsed,
        "peak_gpu_allocated_bytes": max(
            metadata["peak_gpu_allocated_bytes"], torch.cuda.max_memory_allocated()
        ),
        "target_labels_used": config["target_labels_used"],
        "target_validation_labels": 0,
    }
    if stage == "source":
        result.update(
            best_step=metadata["best_step"],
            validation_accuracy=metadata["best_accuracy"],
            source_train_labels=len(support),
            source_validation_labels=len(split["validation"]),
        )
    write_json(output / "result.json", result)
    print(
        f"COMPLETE {method} seed={seed} draw={support_seed} K={k} "
        f"test={result['test']['accuracy']:.4f}",
        flush=True,
    )
    del model, optimizer, scheduler, wave, labels, state
    torch.cuda.empty_cache()
    return result


def zero_shot(data, mel, source, seed, output, device, sig):
    import torch

    if (output / "result.json").exists():
        return read_json(output / "result.json")
    fresh_directory(output)
    test = np.flatnonzero(data["phone_order"] >= 15)
    with np.load(source / "normalization.npz", allow_pickle=False) as stats:
        mean, std = stats["mean"], stats["std"]
    model = initialize_model(
        seed, torch.load(source / "best.pt", map_location="cpu", weights_only=True)
    ).to(device)
    scores = _score(
        model,
        mel[test],
        torch.as_tensor(mean, device=device),
        torch.as_tensor(std, device=device),
        device,
    )
    result = {
        "method": "source_only",
        "k": 0,
        "training_seed": seed,
        "support_seed": None,
        "target_labels_used": 0,
        "target_validation_labels": 0,
        "test": measure(data["phone_y"][test], scores),
        "source_checkpoint_sha256": file_hash(source / "best.pt"),
    }
    write_json(
        output / "config.json",
        {
            "signature": sig,
            "method": "source_only",
            "training_seed": seed,
            "test_ids": data["phone_ids"][test].tolist(),
            "target_labels_used": 0,
            "normalization_fit_on": "source_train",
        },
    )
    np.savez_compressed(output / "scores.npz", scores=scores, y=data["phone_y"][test], indices=test)
    write_json(output / "result.json", result)
    del model
    torch.cuda.empty_cache()
    return result


def run(prepared, cache, output, resume=False):
    configure_rx6000()
    device, hardware = select_device("gpu")
    import torch

    torch.set_num_threads(4)
    sig = signature(prepared, cache)
    if output.exists():
        if not resume:
            raise ResearchError("Direktori sudah ada; gunakan --resume untuk eksperimen yang sama.")
        manifest = read_json(output / "manifest.json")
        if (
            manifest["signature"] != sig
            or manifest["environment"]["packages"] != environment()["packages"]
            or manifest["hardware"]["name"] != hardware["name"]
        ):
            raise ResearchError("Signature, lingkungan atau GPU resume berbeda.")
        if (output / "results.json").exists() and read_json(output / "results.json")[
            "status"
        ] == "complete":
            return read_json(output / "results.json")
    else:
        fresh_directory(output)
        manifest = {
            "created_at": now(),
            "status": "running",
            "signature": sig,
            "hardware": hardware,
            "environment": environment(),
            "blocks": [{"training_seed": seed, "support_seed": draw} for seed, draw in BLOCKS],
        }
        write_json(output / "manifest.json", manifest)
        (output / "protocol.md").write_text(PROTOCOL.read_text())
    data, mel = load_data(prepared, cache)
    source_split = split_indices(data["zoom_y"], data["zoom_order"], seed=42)
    source_split = {**source_split, "support": source_split["train"]}
    sources, conditions = [], []

    def persist(status="running"):
        write_json(
            output / "results.json",
            {
                "status": status,
                "signature": sig,
                "sources": sources,
                "conditions": conditions,
                "updated_at": now(),
            },
        )

    for seed, _ in BLOCKS:
        check_code(sig)
        folder = output / "sources" / f"s{seed}"
        row = fit(
            data, mel["zoom"], "zoom", source_split, seed, None, None, "source", folder, device, sig
        )
        row = {**row, "directory": str(folder.relative_to(output))}
        sources.append(row)
        persist()
        if row["validation_accuracy"] < 0.5:
            persist("failed_source")
            raise ResearchError("Model sumber tidak lulus kelayakan; target tidak dijalankan.")
    lock = {
        "signature": sig,
        "source_checkpoints": {str(row["training_seed"]): row["weights_sha256"] for row in sources},
        "target_updates": UPDATES,
        "target_lr": 5e-4,
        "ks": list(KS),
        "blocks": manifest["blocks"],
    }
    if (output / "sources-locked.json").exists():
        old_lock = read_json(output / "sources-locked.json")
        if any(old_lock[key] != value for key, value in lock.items()):
            raise ResearchError("Checkpoint/aturan sumber berbeda dari kunci sebelum target.")
    else:
        write_json(output / "sources-locked.json", {"created_at": now(), **lock})
    for block, (seed, draw) in enumerate(BLOCKS, start=1):
        source = output / "sources" / f"s{seed}"
        folder = output / "conditions" / f"b{block}-k0"
        row = zero_shot(data, mel["phone"], source, seed, folder, device, sig)
        conditions.append({**row, "block": block, "directory": str(folder.relative_to(output))})
        persist()
        for k in KS:
            split = calibration_split(data["phone_y"], data["phone_order"], k, draw)
            paired = []
            for method in ("scratch", "adapt"):
                check_code(sig)
                folder = output / "conditions" / f"b{block}-k{k}-{method}"
                row = fit(
                    data,
                    mel["phone"],
                    "phone",
                    split,
                    seed,
                    draw,
                    k,
                    method,
                    folder,
                    device,
                    sig,
                    source if method == "adapt" else None,
                )
                paired.append(row)
                conditions.append(
                    {**row, "block": block, "directory": str(folder.relative_to(output))}
                )
                persist()
            if paired[0]["batch_probes"] != paired[1]["batch_probes"]:
                raise ResearchError("Urutan batch/augmentasi pasangan berbeda.")
    check_code(sig)
    persist("complete")
    write_json(output / "manifest.json", {**manifest, "status": "complete", "finished_at": now()})
    print("COMPLETE: 3 source fits, 30 target fits, 33 target evaluations", flush=True)
    return read_json(output / "results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/external/harrison-fewshot")
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "data/external/harrison-fewshot-features"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        run(args.prepared, args.cache, args.output, args.resume)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
