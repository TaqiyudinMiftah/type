"""Fixed-budget source development of a wider stem after attention-only repairs failed.

The training loop is copied unchanged from repair.train_one so the failed
experiments keep their original implementation hash. Model construction and
provenance are local to this module; no runtime patching is used.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

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
from .harrison import augment, measure, spec_mask
from .repair import SEEDS, development_split, input_probe, load_inputs, qualifies
from .repair import repaired_model as attention_model
from .repair import signature as previous_signature

PROTOCOL = ROOT / "research/STEM_REPAIR_PROTOCOL.md"
PARENT = ROOT / "runs/repair-development"
CANDIDATES = ("wide-stem-residual",)


def repaired_model(seed: int, variant: str):
    if variant != "wide-stem-residual":
        return attention_model(seed, variant)
    model, reference = attention_model(seed, "residual-prenorm")
    model[0].s0 = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1, bias=False),
        nn.GroupNorm(8, 32),
        nn.GELU(),
        nn.Conv2d(32, 64, 3, padding=1, bias=False),
        nn.GroupNorm(8, 64),
        nn.GELU(),
    )
    for layer in model[0].s0:
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
    model[0].mlp0 = nn.Identity()
    return model, reference


def signature(prepared, cache):
    parent = read_json(PARENT / "results.json")
    if parent["status"] != "complete" or parent["selected_variant"] is not None:
        raise ResearchError("Tahap awal harus lengkap dan gagal seleksi sebelum pengembangan stem.")
    old_signature = previous_signature(prepared, cache)
    if parent["signature"] != old_signature:
        raise ResearchError("Kode/data/protokol tahap awal berubah.")
    return {
        **old_signature,
        "previous_implementation_sha256": old_signature["implementation_sha256"],
        "previous_protocol_sha256": old_signature["protocol_sha256"],
        "previous_development_sha256": file_hash(PARENT / "results.json"),
        "bn_autopsy_sha256": file_hash(ROOT / "reports/repair-bn-autopsy/results.json"),
        "implementation_sha256": file_hash(Path(__file__)),
        "protocol_sha256": file_hash(PROTOCOL),
    }


def train_one(data, mel, domain, split, seed, variant, updates, output, device, stage):
    """One fixed-budget fit. Evaluation split is read only after the last update."""
    fresh_directory(output)
    support = split["support"]
    evaluation_name = "validation" if stage == "development" else "test"
    evaluation = split[evaluation_name]
    y = data[f"{domain}_y"]
    mean, std = _normalization(mel, support)
    mean_t, std_t = (torch.as_tensor(x, device=device) for x in (mean, std))
    model, reference = repaired_model(seed, variant)
    np.random.seed(seed)
    model.to(device)
    initial_probe = input_probe(model, mel[support], y[support], mean, std, device)
    # Probe is eval-only with no RNG-consuming layers. Reset training RNG explicitly.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    wave = torch.as_tensor(data[f"{domain}_wave"][support], device=device)
    labels = torch.as_tensor(y[support], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1, end_factor=0, total_iters=updates
    )
    config = {
        "stage": stage,
        "domain": domain,
        "variant": variant,
        "training_seed": seed,
        "support_seed": 101,
        "k": 10,
        "updates": updates,
        "batch_size": 16,
        "lr": 5e-4,
        "scheduler": "linear_to_zero",
        "support_ids": data[f"{domain}_ids"][support].tolist(),
        f"{evaluation_name}_ids": data[f"{domain}_ids"][evaluation].tolist(),
        "normalization_fit_on": "support_only",
        "parameters": sum(p.numel() for p in model.parameters()),
        "reference": reference,
        "initial_probe": initial_probe,
    }
    write_json(output / "config.json", config)
    np.savez(output / "normalization.npz", mean=mean, std=std)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    history = []
    for step in range(1, updates + 1):
        model.train()
        ix = torch.randint(len(support), (16,), device=device)
        image = spec_mask(frontend(augment(wave[ix])))
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(((image - mean_t) / std_t)[:, None]), labels[ix])
        if not torch.isfinite(loss):
            raise ResearchError(f"Loss non-finite: {stage}/{variant}/{seed}/{step}")
        loss.backward()
        stem_gradient = None
        if step in (1, 100, 1000, 3000, 6000, 10_000):
            stem_gradient = float(model[0].s0[0].weight.grad.detach().double().norm().item())
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 500 == 0 or stem_gradient is not None:
            row = {"step": step, "loss": float(loss.item())}
            if stem_gradient is not None:
                row["stem_gradient_l2"] = stem_gradient
            if step in (1000, 3000, 6000, updates):
                scores = _score(model, mel[support], mean_t, std_t, device)
                row["support_accuracy"] = float((scores.argmax(1) == y[support]).mean())
                if stage == "development":
                    scores = _score(model, mel[evaluation], mean_t, std_t, device)
                    row["validation_accuracy"] = float((scores.argmax(1) == y[evaluation]).mean())
            history.append(row)
            write_json(output / "history.json", {"updates": history})
            if step == 1 or step % 1000 == 0:
                print(
                    f"{stage} {variant} seed={seed} {step}/{updates} "
                    f"loss={row['loss']:.4f} support={row.get('support_accuracy')} "
                    f"validation={row.get('validation_accuracy')}",
                    flush=True,
                )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    support_scores = _score(model, mel[support], mean_t, std_t, device)
    scores = _score(model, mel[evaluation], mean_t, std_t, device)
    result = {
        "stage": stage,
        "domain": domain,
        "variant": variant,
        "training_seed": seed,
        "support_accuracy": float((support_scores.argmax(1) == y[support]).mean()),
        "support_max_logit_range": float(np.ptp(support_scores, axis=0).max()),
        "initial_probe": initial_probe,
        "final_probe": input_probe(model, mel[support], y[support], mean, std, device),
        evaluation_name: measure(y[evaluation], scores),
        "updates": updates,
        "training_seconds": elapsed,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "directory": output.name,
    }
    torch.save(
        {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        output / "last.pt",
    )
    np.savez_compressed(
        output / "scores.npz",
        scores=scores,
        y=y[evaluation],
        indices=evaluation,
        support=support,
        support_scores=support_scores,
    )
    result["weights_sha256"] = file_hash(output / "last.pt")
    write_json(output / "result.json", result)
    print(
        f"COMPLETE {variant} seed={seed} "
        f"{evaluation_name}={result[evaluation_name]['accuracy']:.4f}",
        flush=True,
    )
    del model, optimizer, scheduler, wave, labels
    torch.cuda.empty_cache()
    return result


def development(prepared, cache, output):
    configure_rx6000()
    device, hardware = select_device("gpu")
    torch.set_num_threads(4)
    sig = signature(prepared, cache)
    fresh_directory(output)
    manifest = {
        "created_at": now(),
        "signature": sig,
        "hardware": hardware,
        "environment": environment(),
    }
    write_json(output / "manifest.json", manifest)
    (output / "protocol.md").write_text(PROTOCOL.read_text())
    data, features = load_inputs(prepared, cache)
    split = development_split(data["zoom_y"], data["zoom_order"])
    initial_probes = []
    for seed in SEEDS:
        mean, std = _normalization(features["zoom_mel"], split["support"])
        for variant in ("original", "leaky", *CANDIDATES):
            model, _ = repaired_model(seed, variant)
            model.to(device)
            probe = input_probe(
                model,
                features["zoom_mel"][split["support"]],
                data["zoom_y"][split["support"]],
                mean,
                std,
                device,
            )
            initial_probes.append({"variant": variant, "seed": seed, **probe})
            del model
    write_json(
        output / "initial-probes.json",
        {"domain": "zoom", "target_used": False, "probes": initial_probes},
    )
    rows = []
    selected = None
    for variant in CANDIDATES:
        candidate_rows = []
        for seed in SEEDS:
            result = train_one(
                data,
                features["zoom_mel"],
                "zoom",
                split,
                seed,
                variant,
                6000,
                output / f"{variant}-s{seed}",
                device,
                "development",
            )
            candidate_rows.append(result)
            rows.append(result)
            write_json(
                output / "results.json", {"status": "running", "signature": sig, "conditions": rows}
            )
        if qualifies(candidate_rows):
            selected = variant
            break
    result = {
        "status": "complete",
        "signature": sig,
        "conditions": rows,
        "selected_variant": selected,
        "finished_at": now(),
    }
    write_json(output / "results.json", result)
    decision = {
        "created_at": now(),
        "status": "locked" if selected else "failed_development",
        "variant": selected,
        "signature": sig,
        "development_results_sha256": file_hash(output / "results.json"),
        "selection_domain": "zoom",
        "phone_labels_used_for_selection": 0,
        "source_fit_labels": 360,
        "source_validation_labels": 72,
        "target_updates": 10_000,
        "target_training_seeds": list(SEEDS),
        "target_support_seed": 101,
        "target_k": 10,
    }
    write_json(output / "decision.json", decision)
    print(f"DEVELOPMENT COMPLETE selected={selected}", flush=True)
    return result


def confirm(prepared, cache, development_dir, output):
    sig = signature(prepared, cache)
    decision = read_json(development_dir / "decision.json")
    dev = read_json(development_dir / "results.json")
    if (
        decision["status"] != "locked"
        or decision["signature"] != sig
        or decision["development_results_sha256"] != file_hash(development_dir / "results.json")
    ):
        raise ResearchError("Keputusan pengembangan belum terkunci atau input/kode berubah.")
    selected = decision["variant"]
    if not qualifies([r for r in dev["conditions"] if r["variant"] == selected]):
        raise ResearchError("Kandidat tidak memenuhi kriteria pengembangan.")
    configure_rx6000()
    device, hardware = select_device("gpu")
    torch.set_num_threads(4)
    fresh_directory(output)
    manifest = {
        "created_at": now(),
        "signature": sig,
        "decision_sha256": file_hash(development_dir / "decision.json"),
        "hardware": hardware,
        "environment": environment(),
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "decision.json", decision)
    data, features = load_inputs(prepared, cache)
    split = calibration_split(data["phone_y"], data["phone_order"], 10, 101)
    for old_dir in (
        ROOT / "runs/fewshot-optimization-diagnostic/phone",
        ROOT / "runs/stability-gate/leaky-s42",
    ):
        config = read_json(old_dir / "config.json")
        if (
            config["support_ids"] != data["phone_ids"][split["support"]].tolist()
            or config["test_ids"] != data["phone_ids"][split["test"]].tolist()
        ):
            raise ResearchError("Identitas dukungan/uji berbeda dari kontrol lama.")
    rows = []
    for seed in SEEDS:
        rows.append(
            train_one(
                data,
                features["phone_mel"],
                "phone",
                split,
                seed,
                selected,
                10_000,
                output / f"{selected}-s{seed}",
                device,
                "confirmation",
            )
        )
        write_json(
            output / "results.json", {"status": "running", "signature": sig, "conditions": rows}
        )
    result = {
        "status": "complete",
        "signature": sig,
        "decision_sha256": manifest["decision_sha256"],
        "conditions": rows,
        "finished_at": now(),
    }
    write_json(output / "results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("develop", "confirm"))
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/external/harrison-fewshot")
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "data/external/harrison-fewshot-features"
    )
    parser.add_argument("--development", type=Path, default=ROOT / "runs/stem-repair-development")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.stage == "develop":
            development(args.prepared, args.cache, args.output)
        else:
            confirm(args.prepared, args.cache, args.development, args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
