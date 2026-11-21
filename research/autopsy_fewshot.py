"""Support-only activation/gradient inspection of the saved collapsed model."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from ketik_audio.common import ResearchError, file_hash, fresh_directory, read_json, write_json
from ketik_audio.fewshot import ROOT, calibration_split
from ketik_audio.frozen import assert_unchanged, state_snapshot
from ketik_audio.harrison import _reference_model

STAGES = [
    "0.s0.0",
    "0.s0.1",
    "0.s0.2",
    "0.mlp0.0",
    "0.mlp0.1",
    "0.mlp0.2",
    "0.s1",
    "0.mlp1.0",
    "0.mlp1.1",
    "0.mlp1.2",
    "0.s2",
    "0.mlp2.0",
    "0.mlp2.1",
    "0.mlp2.2",
    "0.s3",
    "0.mlp3.0",
    "0.mlp3.1",
    "0.mlp3.2",
    "0.s4",
    "0.mlp4.0",
    "0.mlp4.1",
    "0.mlp4.2",
    "1",
    "2",
    "3",
]


def stages(model, images):
    model.eval()
    modules = dict(model.named_modules())
    stats, references, handles = {}, {}, []

    def hook(name, module, args, output):
        value = output.detach()
        source = args[0].detach()
        if name not in stats:
            references[name] = value[0].clone()
            stats[name] = {
                "name": name,
                "type": type(module).__name__,
                "shape_per_event": list(value.shape[1:]),
                "numel": 0,
                "positive": 0,
                "min": float("inf"),
                "max": float("-inf"),
                "input_min": float("inf"),
                "input_max": float("-inf"),
                "max_difference_from_first_event": 0.0,
            }
        row = stats[name]
        row["numel"] += value.numel()
        row["positive"] += int((value > 0).sum())
        row["min"], row["max"] = (
            min(row["min"], float(value.min())),
            max(row["max"], float(value.max())),
        )
        row["input_min"] = min(row["input_min"], float(source.min()))
        row["input_max"] = max(row["input_max"], float(source.max()))
        row["max_difference_from_first_event"] = max(
            row["max_difference_from_first_event"], float((value - references[name]).abs().max())
        )

    for name in STAGES:
        handles.append(
            modules[name].register_forward_hook(
                lambda module, args, output, name=name: hook(name, module, args, output)
            )
        )
    scores = []
    with torch.inference_mode():
        for start in range(0, len(images), 32):
            scores.append(model(images[start : start + 32]).numpy())
    for handle in handles:
        handle.remove()
    for row in stats.values():
        row["positive_fraction"] = row.pop("positive") / row.pop("numel")
    return list(stats.values()), np.concatenate(scores)


def gradient_probe(model, images, labels, training):
    model = copy.deepcopy(model)
    model.requires_grad_(True).train(training)
    before = state_snapshot(model)
    x = images.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    groups = {}
    for name, parameter in model.named_parameters():
        group = ".".join(name.split(".")[:2]) if name.startswith("0.") else name.split(".")[0]
        row = groups.setdefault(group, {"squared_l2": 0.0, "nonzero_values": 0, "values": 0})
        if parameter.grad is not None:
            row["squared_l2"] += float(parameter.grad.double().square().sum())
            row["nonzero_values"] += int(torch.count_nonzero(parameter.grad))
            row["values"] += parameter.grad.numel()
    for row in groups.values():
        row["gradient_l2"] = row.pop("squared_l2") ** 0.5
    changed = [
        key for key, value in model.state_dict().items() if not torch.equal(before[key], value)
    ]
    if not training and changed:
        raise ResearchError("State berubah pada gradient probe eval.")
    return {
        "mode": "train-copy" if training else "eval-copy",
        "loss": float(loss.detach()),
        "input_gradient_l2": float(x.grad.double().norm()),
        "max_logit_range": float(
            (logits.max(dim=0).values - logits.min(dim=0).values).max().detach()
        ),
        "gradient_groups": groups,
        "changed_buffers_discarded": changed,
        "optimizer_steps": 0,
    }


def run(primary: Path, diagnostic: Path, prepared: Path, cache: Path, output: Path):
    torch.set_num_threads(4)
    original = read_json(primary / "results.json")
    diag = read_json(diagnostic / "results.json")
    if (
        original["status"] != "complete"
        or diag["status"] != "complete"
        or file_hash(prepared / "segments.npz") != original["signature"]["segments_sha256"]
        or file_hash(cache / "features.npz") != original["signature"]["features_sha256"]
    ):
        raise ResearchError("Input autopsi tidak sesuai pilot selesai.")
    fresh_directory(output)
    with np.load(prepared / "segments.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    with np.load(cache / "features.npz", allow_pickle=False) as archive:
        mel = {s: archive[f"{s}_mel"] for s in ["phone", "zoom"]}
    cases = [
        (
            "collapsed-phone",
            "phone",
            diagnostic / "phone/last.pt",
            diagnostic / "phone/normalization.npz",
        ),
        (
            "trained-zoom",
            "zoom",
            diagnostic / "zoom/last.pt",
            diagnostic / "zoom/normalization.npz",
        ),
        ("initial-phone", "phone", None, diagnostic / "phone/normalization.npz"),
        (
            "source-phone",
            "phone",
            primary / "source-phone/best.pt",
            primary / "source-phone/normalization.npz",
        ),
    ]
    records = []
    for name, domain, weights, normalization in cases:
        torch.manual_seed(42)
        model, _ = _reference_model()
        if weights:
            model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        before = state_snapshot(model)
        split = calibration_split(data[f"{domain}_y"], data[f"{domain}_order"], 10, 101)
        support = split["support"]
        with np.load(normalization, allow_pickle=False) as stats:
            images = torch.from_numpy(
                ((mel[domain][support] - stats["mean"]) / stats["std"])[:, None]
            )
        activations, scores = stages(model, images)
        balanced = np.arange(36) * 10
        labels = torch.from_numpy(data[f"{domain}_y"][support[balanced]])
        probes = [
            gradient_probe(model, images[balanced], labels, training) for training in [False, True]
        ]
        dead = [
            row
            for row in activations
            if row["type"] == "ReLU" and row["max"] == 0 and row["input_max"] > row["input_min"]
        ]
        intervention = None
        if name == "collapsed-phone" and dead:
            selected = dead[0]["name"]
            modified = copy.deepcopy(model)
            parent_name, child_name = selected.rsplit(".", 1)
            parent = modified.get_submodule(parent_name)
            parent._modules[child_name] = torch.nn.LeakyReLU(0.01)
            changed_activations, changed_scores = stages(modified, images)
            intervention = {
                "selected_from_support_only": selected,
                "replacement": "LeakyReLU(0.01)",
                "optimizer_steps": 0,
                "max_logit_range": float(np.ptp(changed_scores, axis=0).max()),
                "gradient": gradient_probe(modified, images[balanced], labels, False),
                "activations": changed_activations,
            }
        assert_unchanged(model, before)
        row = {
            "case": name,
            "domain": domain,
            "weights_sha256": file_hash(weights) if weights else None,
            "normalization_sha256": file_hash(normalization),
            "support_ids": data[f"{domain}_ids"][support].tolist(),
            "target_test_used": False,
            "original_state_unchanged": True,
            "max_logit_range": float(np.ptp(scores, axis=0).max()),
            "distinct_predictions_on_support": np.unique(scores.argmax(axis=1)).tolist(),
            "activations": activations,
            "gradient_probes": probes,
            "dead_relu_candidates": dead,
            "local_intervention": intervention,
        }
        write_json(output / f"{name}.json", row)
        records.append(row)
        print(
            f"Autopsy {name}: logit range={row['max_logit_range']:.6g}, "
            f"dead ReLU={[r['name'] for r in dead]}",
            flush=True,
        )
    result = {
        "cases": records,
        "device": "cpu",
        "protocol_sha256": file_hash(ROOT / "research/FROZEN_HEAD_PROTOCOL.md"),
        "implementation_sha256": file_hash(Path(__file__)),
        "primary_results_sha256": file_hash(primary / "results.json"),
        "diagnostic_results_sha256": file_hash(diagnostic / "results.json"),
        "optimizer_steps": 0,
    }
    write_json(output / "results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.primary, args.diagnostic, args.prepared, args.cache, args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
