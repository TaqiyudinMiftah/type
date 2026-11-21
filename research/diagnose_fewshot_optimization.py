"""A separately specified, post-result scratch optimization diagnostic."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ketik_audio.common import (
    ResearchError,
    environment,
    file_hash,
    fresh_directory,
    read_json,
    write_json,
)
from ketik_audio.devices import configure_rx6000, select_device
from ketik_audio.fewshot import _normalization, _score, _seed, calibration_split, frontend
from ketik_audio.harrison import _reference_model, augment, measure, spec_mask

PROTOCOL = Path(__file__).resolve().parent / "FEWSHOT_DIAGNOSTIC_PROTOCOL.md"


def diagnose(primary: Path, prepared: Path, cache: Path, output: Path):
    configure_rx6000()
    device, hardware = select_device("gpu")
    import torch

    torch.set_num_threads(4)
    record = read_json(primary / "results.json")
    if record["status"] != "complete":
        raise ResearchError("Selesaikan seluruh pilot utama dahulu.")
    signature = {
        "primary_results_sha256": file_hash(primary / "results.json"),
        "protocol_sha256": file_hash(PROTOCOL),
        "implementation_sha256": file_hash(Path(__file__)),
    }
    if file_hash(prepared / "segments.npz") != record["signature"]["segments_sha256"]:
        raise ResearchError("Potongan berbeda dari pilot utama.")
    if file_hash(cache / "features.npz") != record["signature"]["features_sha256"]:
        raise ResearchError("Fitur berbeda dari pilot utama.")
    if output.exists():
        if read_json(output / "manifest.json")["signature"] != signature:
            raise ResearchError("Tidak dapat resume diagnostik dengan input/kode berbeda.")
    else:
        fresh_directory(output)
        write_json(
            output / "manifest.json",
            {"signature": signature, "hardware": hardware, "environment": environment()},
        )
        (output / "protocol.md").write_text(PROTOCOL.read_text())
    data_file = np.load(prepared / "segments.npz", allow_pickle=False)
    feature_file = np.load(cache / "features.npz", allow_pickle=False)
    data = {key: data_file[key] for key in data_file.files}
    features = {key: feature_file[key] for key in feature_file.files}
    results = []
    for source, target in [("phone", "zoom"), ("zoom", "phone")]:
        out = output / target
        if (out / "result.json").exists():
            results.append(read_json(out / "result.json"))
            continue
        out.mkdir(exist_ok=True)
        _seed()
        y = data[f"{target}_y"]
        split = calibration_split(y, data[f"{target}_order"], 10, 101)
        support, test = split["support"], split["test"]
        mean, std = _normalization(features[f"{target}_mel"], support)
        mean_t, std_t = (torch.as_tensor(x, device=device) for x in [mean, std])
        wave = torch.as_tensor(data[f"{target}_wave"][support], device=device)
        labels = torch.as_tensor(y[support], device=device)
        model, _ = _reference_model()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=0, total_iters=10_000
        )
        write_json(
            out / "config.json",
            {
                "target": target,
                "k": 10,
                "support_seed": 101,
                "training_seed": 42,
                "updates": 10_000,
                "lr": 5e-4,
                "target_labels_used": 360,
                "target_validation_labels": 0,
                "support_ids": data[f"{target}_ids"][support].tolist(),
                "test_ids": data[f"{target}_ids"][test].tolist(),
                "protocol_sha256": file_hash(PROTOCOL),
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
                raise ResearchError("Loss diagnostik non-finite.")
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step == 1 or step % 500 == 0:
                row = {"step": step, "loss": float(loss.item())}
                if step in [1000, 3000, 6000, 10_000]:
                    scores = _score(
                        model, features[f"{target}_mel"][support], mean_t, std_t, device
                    )
                    row["support_accuracy"] = float(np.mean(scores.argmax(axis=1) == y[support]))
                history.append(row)
                write_json(out / "history.json", {"updates": history})
                print(
                    f"diagnostic scratch {target}: update={step}/10000 "
                    f"loss={row['loss']:.4f} support={row.get('support_accuracy')}",
                    flush=True,
                )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        scores = _score(model, features[f"{target}_mel"][test], mean_t, std_t, device)
        result = {
            "source": source,
            "target": target,
            "k": 10,
            "support_seed": 101,
            "updates": 10_000,
            "target_labels_used": 360,
            "training_seconds": elapsed,
            "support_accuracy": history[-1]["support_accuracy"],
            "test": measure(y[test], scores),
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        torch.save(
            {key: v.detach().cpu().clone() for key, v in model.state_dict().items()},
            out / "last.pt",
        )
        np.savez_compressed(
            out / "scores.npz", scores=scores, y=y[test], indices=test, support=support
        )
        write_json(out / "result.json", result)
        results.append(result)
        del model, optimizer, scheduler, wave, labels
        torch.cuda.empty_cache()
    write_json(
        output / "results.json",
        {"status": "complete", "signature": signature, "conditions": results},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        diagnose(args.primary, args.prepared, args.cache, args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
