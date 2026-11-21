"""Source-only BN recalibration diagnostic on copies of two repair checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ketik_audio.common import file_hash, fresh_directory, write_json
from ketik_audio.fewshot import ROOT, _normalization, _score
from ketik_audio.repair import development_split, load_inputs, repaired_model


def run(output):
    torch.set_num_threads(4)
    fresh_directory(output)
    data, features = load_inputs(
        ROOT / "data/external/harrison-fewshot", ROOT / "data/external/harrison-fewshot-features"
    )
    y, mel = data["zoom_y"], features["zoom_mel"]
    split = development_split(y, data["zoom_order"])
    support, val = split["support"], split["validation"]
    mean, std = _normalization(mel, support)
    rows = []
    for seed in (42, 43):
        folder = ROOT / f"runs/repair-development/residual-prenorm-s{seed}"
        digest = file_hash(folder / "last.pt")
        model, _ = repaired_model(seed, "residual-prenorm")
        model.load_state_dict(torch.load(folder / "last.pt", map_location="cpu", weights_only=True))
        snapshot = {k: v.clone() for k, v in model.state_dict().items()}
        before = _score(model, mel[val], torch.from_numpy(mean), torch.from_numpy(std), "cpu")
        model.eval()
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.reset_running_stats()
                module.momentum = None
                module.train()
        with torch.inference_mode():
            for batch in support.reshape(36, 10).T:
                model(torch.from_numpy(((mel[batch] - mean) / std)[:, None]))
        after = _score(model, mel[val], torch.from_numpy(mean), torch.from_numpy(std), "cpu")
        trained = _score(model, mel[support], torch.from_numpy(mean), torch.from_numpy(std), "cpu")
        changed = [k for k, v in model.state_dict().items() if not torch.equal(v, snapshot[k])]
        assert all("running_" in k or "num_batches_tracked" in k for k in changed)
        assert digest == file_hash(folder / "last.pt")
        row = {
            "seed": seed,
            "weights_sha256": digest,
            "before_validation": float((before.argmax(1) == y[val]).mean()),
            "recalibrated_validation": float((after.argmax(1) == y[val]).mean()),
            "recalibrated_support": float((trained.argmax(1) == y[support]).mean()),
            "changed_buffers": changed,
            "target_data_used": False,
            "optimizer_updates": 0,
        }
        rows.append(row)
        print(row, flush=True)
    write_json(
        output / "results.json",
        {
            "diagnostic": "source-only-clean-BN-recalibration",
            "calibration_examples": 360,
            "validation_examples": 72,
            "conditions": rows,
            "original_checkpoints_unchanged": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
