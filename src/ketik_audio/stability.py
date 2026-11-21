"""A fixed, three-initialization diagnostic of the observed final ReLU gate."""

from __future__ import annotations

import argparse
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
from .fewshot import ROOT, _normalization, _score, calibration_split, frontend
from .harrison import _reference_model, augment, measure, spec_mask

PROTOCOL = ROOT / "research/STABILITY_PROTOCOL.md"


def variant_model(seed, variant):
    import torch

    if variant not in ["relu", "leaky"]:
        raise ResearchError("Varian aktivasi tidak dikenal.")
    torch.manual_seed(seed)
    model, reference = _reference_model()
    if not isinstance(model[0].mlp4[1], torch.nn.ReLU):
        raise ResearchError("Arsitektur referensi berubah; lokasi intervensi perlu diaudit ulang.")
    if variant == "leaky":
        model[0].mlp4[1] = torch.nn.LeakyReLU(negative_slope=0.01)
    return model, reference


def run(primary: Path, diagnostic: Path, prepared: Path, cache: Path, autopsy: Path, output: Path):
    configure_rx6000()
    device, hardware = select_device("gpu")
    import torch

    torch.set_num_threads(4)
    original, prior = read_json(primary / "results.json"), read_json(diagnostic / "results.json")
    if original["status"] != "complete" or prior["status"] != "complete":
        raise ResearchError("Hasil kontrol lama belum lengkap.")
    if (
        file_hash(prepared / "segments.npz") != original["signature"]["segments_sha256"]
        or file_hash(cache / "features.npz") != original["signature"]["features_sha256"]
    ):
        raise ResearchError("Input berbeda dari pilot.")
    autopsy_record = read_json(autopsy / "results.json")
    failed_case = next(
        case for case in autopsy_record["cases"] if case["case"] == "collapsed-phone"
    )
    if not any(gate["name"] == "0.mlp4.1" for gate in failed_case["dead_relu_candidates"]):
        raise ResearchError("Autopsi tidak mengonfirmasi lokasi aktivasi yang akan diubah.")
    fresh_directory(output)
    signature = {
        "primary_results_sha256": file_hash(primary / "results.json"),
        "old_diagnostic_sha256": file_hash(diagnostic / "results.json"),
        "autopsy_sha256": file_hash(autopsy / "results.json"),
        "protocol_sha256": file_hash(PROTOCOL),
        "implementation_sha256": file_hash(Path(__file__)),
        "segments_sha256": original["signature"]["segments_sha256"],
        "features_sha256": original["signature"]["features_sha256"],
    }
    write_json(
        output / "manifest.json",
        {
            "created_at": now(),
            "status": "running",
            "signature": signature,
            "hardware": hardware,
            "environment": environment(),
        },
    )
    (output / "protocol.md").write_text(PROTOCOL.read_text())
    with np.load(prepared / "segments.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    with np.load(cache / "features.npz", allow_pickle=False) as archive:
        mel = archive["phone_mel"]
    y = data["phone_y"]
    split = calibration_split(y, data["phone_order"], 10, 101)
    support, test = split["support"], split["test"]
    old_config = read_json(diagnostic / "phone/config.json")
    if (
        old_config["support_ids"] != data["phone_ids"][support].tolist()
        or old_config["test_ids"] != data["phone_ids"][test].tolist()
        or old_config["updates"] != 10_000
        or old_config["training_seed"] != 42
        or old_config["lr"] != 5e-4
    ):
        raise ResearchError("Kontrol lama tidak cocok.")
    old = read_json(diagnostic / "phone/result.json")
    conditions = [
        {
            **old,
            "variant": "relu",
            "training_seed": 42,
            "reused_directory": str((diagnostic / "phone").resolve()),
            "weights_sha256": file_hash(diagnostic / "phone/last.pt"),
        }
    ]
    mean, std = _normalization(mel, support)
    with np.load(diagnostic / "phone/normalization.npz", allow_pickle=False) as stats:
        if not (np.array_equal(mean, stats["mean"]) and np.array_equal(std, stats["std"])):
            raise ResearchError("Statistik kontrol lama berbeda.")
    mean_t, std_t = (torch.as_tensor(x, device=device) for x in [mean, std])
    for variant, seed in [("relu", 43), ("relu", 44), ("leaky", 42), ("leaky", 43), ("leaky", 44)]:
        np.random.seed(seed)
        model, reference = variant_model(seed, variant)
        torch.cuda.manual_seed_all(seed)
        model.to(device)
        out = output / f"{variant}-s{seed}"
        out.mkdir()
        wave = torch.as_tensor(data["phone_wave"][support], device=device)
        labels = torch.as_tensor(y[support], device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=0, total_iters=10_000
        )
        write_json(
            out / "config.json",
            {
                "variant": variant,
                "training_seed": seed,
                "support_seed": 101,
                "k": 10,
                "updates": 10_000,
                "target_labels_used": 360,
                "target_validation_labels": 0,
                "batch_size": 16,
                "lr": 5e-4,
                "support_ids": data["phone_ids"][support].tolist(),
                "test_ids": data["phone_ids"][test].tolist(),
                "reference": reference,
                "protocol_sha256": signature["protocol_sha256"],
            },
        )
        np.savez(out / "normalization.npz", mean=mean, std=std)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        history = []
        for step in range(1, 10_001):
            model.train()
            ix = torch.randint(len(support), (16,), device=device)
            image = spec_mask(frontend(augment(wave[ix])))
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(
                model(((image - mean_t) / std_t)[:, None]), labels[ix]
            )
            if not torch.isfinite(loss):
                raise ResearchError(f"Loss non-finite: {variant}/{seed}/{step}")
            loss.backward()
            gradient = None
            if step in [1, 100, 1000, 3000, 6000, 10_000]:
                gradient = float(model[0].mlp4[0].weight.grad.detach().double().norm().item())
            optimizer.step()
            scheduler.step()
            if step == 1 or step % 500 == 0 or gradient is not None:
                row = {"step": step, "loss": float(loss.item())}
                if gradient is not None:
                    row["pre_gate_weight_gradient_l2"] = gradient
                if step in [1000, 3000, 6000, 10_000]:
                    scores = _score(model, mel[support], mean_t, std_t, device)
                    row["support_accuracy"] = float(np.mean(scores.argmax(axis=1) == y[support]))
                history.append(row)
                write_json(out / "history.json", {"updates": history})
                if step == 1 or step % 1000 == 0:
                    print(
                        f"{variant} seed={seed} update={step}/10000 loss={row['loss']:.4f} "
                        f"support={row.get('support_accuracy')}",
                        flush=True,
                    )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        support_scores = _score(model, mel[support], mean_t, std_t, device)
        scores = _score(model, mel[test], mean_t, std_t, device)
        row = {
            "variant": variant,
            "training_seed": seed,
            "support_seed": 101,
            "k": 10,
            "target": "phone",
            "updates": 10_000,
            "target_labels_used": 360,
            "support_accuracy": float(np.mean(support_scores.argmax(axis=1) == y[support])),
            "support_max_logit_range": float(np.ptp(support_scores, axis=0).max()),
            "test": measure(y[test], scores),
            "training_seconds": elapsed,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "directory": out.name,
        }
        torch.save(
            {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            out / "last.pt",
        )
        np.savez_compressed(
            out / "scores.npz", scores=scores, y=y[test], indices=test, support=support
        )
        write_json(out / "result.json", row)
        conditions.append(row)
        write_json(
            output / "results.json",
            {"status": "running", "signature": signature, "conditions": conditions},
        )
        print(f"COMPLETE {variant} seed={seed} test={row['test']['accuracy']:.4f}", flush=True)
        del model, optimizer, scheduler, wave, labels
        torch.cuda.empty_cache()
    result = {
        "status": "complete",
        "signature": signature,
        "conditions": conditions,
        "finished_at": now(),
    }
    write_json(output / "results.json", result)
    manifest = read_json(output / "manifest.json")
    write_json(output / "manifest.json", {**manifest, "status": "complete", "finished_at": now()})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--autopsy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.primary, args.diagnostic, args.prepared, args.cache, args.autopsy, args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
