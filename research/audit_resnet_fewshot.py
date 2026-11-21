"""Audit label budgets, initialization, batch pairs and all CNN checkpoints on CPU."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, read_json, write_json
from ketik_audio.fewshot import KS, ROOT, _normalization, _score, calibration_split
from ketik_audio.harrison import KEYS, measure, split_indices
from ketik_audio.resnet_fewshot import (
    BLOCKS,
    PROBE_STEPS,
    UPDATES,
    initialize_model,
    load_data,
    signature,
    state_digest,
)


def verify_recovery(folder, initial):
    saved = torch.load(folder / "recovery.pt", map_location="cpu", weights_only=True)
    last = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
    if saved["step"] != UPDATES or state_digest(saved["model"]) != state_digest(last):
        raise ResearchError("Checkpoint akhir/pemulihan tidak konsisten.")
    optimizer, scheduler = saved["optimizer"], saved["scheduler"]
    if not optimizer["state"] or any(
        int(v["step"].item()) != UPDATES for v in optimizer["state"].values()
    ):
        raise ResearchError("Counter Adam bukan 10.000 update.")
    if scheduler["last_epoch"] != UPDATES or any(g["lr"] != 0 for g in optimizer["param_groups"]):
        raise ResearchError("Jadwal LR tidak selesai.")
    for name, value in last.items():
        if (
            name.endswith("num_batches_tracked")
            and int(value.item()) - int(initial[name].item()) != UPDATES
        ):
            raise ResearchError("BatchNorm tidak mengikuti tepat 10.000 update.")
    return saved


def replay(folder, row, initial, data, mel, domain, split, sig, predictions):
    cfg = read_json(folder / "config.json")
    support, test = split["support"], split["test"]
    ids, y = data[f"{domain}_ids"], data[f"{domain}_y"]
    if (
        cfg["signature"] != sig
        or cfg["support_ids"] != ids[support].tolist()
        or cfg["test_ids"] != ids[test].tolist()
    ):
        raise ResearchError("Signature atau ID fit/uji berubah.")
    if (
        cfg["updates"] != UPDATES
        or cfg["batch_size"] != 16
        or cfg["lr"] != 5e-4
        or cfg["scheduler"] != "linear_to_zero"
        or cfg["optimizer_initial_state_entries"] != 0
        or cfg["target_validation_labels"] != 0
    ):
        raise ResearchError("Optimizer/anggaran target berbeda.")
    if cfg["initial_state_digest"] != state_digest(initial) or row[
        "initial_state_digest"
    ] != state_digest(initial):
        raise ResearchError("Inisialisasi tidak menyalin model sumber/acak yang ditetapkan.")
    mean, std = _normalization(mel, support)
    with np.load(folder / "normalization.npz", allow_pickle=False) as stats:
        if not np.array_equal(mean, stats["mean"]) or not np.array_equal(std, stats["std"]):
            raise ResearchError("Statistik input bukan dari dukungan yang ditetapkan.")
    if (
        file_hash(folder / "normalization.npz") != row["normalization_sha256"]
        or file_hash(folder / row["checkpoint"]) != row["weights_sha256"]
    ):
        raise ResearchError("Checkpoint/statistik input berubah.")
    weights = torch.load(folder / row["checkpoint"], map_location="cpu", weights_only=True)
    model = initialize_model(row["training_seed"], weights)
    scores = _score(model, mel[test], torch.from_numpy(mean), torch.from_numpy(std), "cpu")
    fitted = _score(model, mel[support], torch.from_numpy(mean), torch.from_numpy(std), "cpu")
    metrics = measure(y[test], scores)
    if (
        metrics != row["test"]
        or float((fitted.argmax(1) == y[support]).mean()) != row["support_accuracy"]
    ):
        raise ResearchError(f"Replay CPU berbeda: {folder}")
    with np.load(folder / "scores.npz", allow_pickle=False) as saved:
        if not (
            np.array_equal(saved["indices"], test)
            and np.array_equal(saved["support"], support)
            and np.array_equal(saved["y"], y[test])
        ):
            raise ResearchError("Indeks/label skor berbeda.")
        if not np.array_equal(saved["scores"].argmax(1), scores.argmax(1)) or not np.array_equal(
            saved["support_scores"].argmax(1), fitted.argmax(1)
        ):
            raise ResearchError("Prediksi individual CPU berbeda.")
        if not np.allclose(saved["scores"], scores, rtol=1e-3, atol=1e-3):
            raise ResearchError("Selisih skor CPU/GPU terlalu besar.")
        maximum_difference = float(np.abs(saved["scores"] - scores).max())
    recovery = verify_recovery(folder, initial)
    history = read_json(folder / "history.json")["updates"]
    if history != recovery["metadata"]["history"] or history[-1]["step"] != UPDATES:
        raise ResearchError("History tidak sama dengan checkpoint pemulihan.")
    if row["batch_probes"] != recovery["metadata"]["batch_probes"] or [
        p["step"] for p in row["batch_probes"]
    ] != list(PROBE_STEPS):
        raise ResearchError("Fingerprint batch tidak lengkap/berbeda.")
    if row["method"] == "source":
        validation = split["validation"]
        if (
            cfg["validation_ids"] != ids[validation].tolist()
            or cfg["source_train_labels"] != 648
            or cfg["source_validation_labels"] != 72
        ):
            raise ResearchError("Anggaran sumber berbeda.")
        validated = [h for h in history if "validation_accuracy" in h]
        best_accuracy = max(h["validation_accuracy"] for h in validated)
        best_step = next(h["step"] for h in validated if h["validation_accuracy"] == best_accuracy)
        if row["best_step"] != best_step or row["validation_accuracy"] != best_accuracy:
            raise ResearchError("Checkpoint sumber bukan maksimum validasi paling awal.")
        scores_val = _score(
            model, mel[validation], torch.from_numpy(mean), torch.from_numpy(std), "cpu"
        )
        if float((scores_val.argmax(1) == y[validation]).mean()) != best_accuracy:
            raise ResearchError("Validasi checkpoint sumber berbeda.")
        if state_digest(weights) != state_digest(recovery["metadata"]["best_state"]):
            raise ResearchError("Bobot sumber tidak cocok dengan snapshot terbaik.")
    else:
        if (
            cfg["target_labels_used"] != 36 * row["k"]
            or row["target_labels_used"] != 36 * row["k"]
            or cfg["selection"] != "last_update"
            or row["checkpoint"] != "last.pt"
        ):
            raise ResearchError("Anggaran/pemilihan checkpoint target berbeda.")
        if any("validation_accuracy" in h or "test_accuracy" in h for h in history):
            raise ResearchError("Target dievaluasi untuk memilih checkpoint.")
    for index, predicted in zip(test, scores.argmax(1), strict=True):
        predictions.append(
            {
                "domain": domain,
                "method": row["method"],
                "training_seed": row["training_seed"],
                "support_seed": row["support_seed"],
                "k": row["k"],
                "event_id": str(ids[index]),
                "actual": KEYS[y[index]],
                "predicted": KEYS[predicted],
                "actual_index": int(y[index]),
                "predicted_index": int(predicted),
            }
        )
    return {
        "directory": str(folder.relative_to(ROOT)) if folder.is_relative_to(ROOT) else str(folder),
        "method": row["method"],
        "training_seed": row["training_seed"],
        "k": row["k"],
        "confusion_match": True,
        "support_predictions_match": True,
        "initialization_match": True,
        "optimizer_updates_match": True,
        "max_absolute_score_difference": maximum_difference,
    }


def run(root, prepared, cache, output):
    root, prepared, cache, output = (p.resolve() for p in (root, prepared, cache, output))
    torch.set_num_threads(4)
    record, manifest = read_json(root / "results.json"), read_json(root / "manifest.json")
    sig = signature(prepared, cache)
    if (
        record["status"] != "complete"
        or manifest["status"] != "complete"
        or record["signature"] != sig
        or manifest["signature"] != sig
    ):
        raise ResearchError("Eksperimen belum lengkap atau signature berubah.")
    if file_hash(root / "protocol.md") != sig["protocol_sha256"]:
        raise ResearchError("Salinan protokol berubah.")
    expected = {
        (block, method, k, seed, draw)
        for block, (seed, draw) in enumerate(BLOCKS, 1)
        for method in ("scratch", "adapt")
        for k in KS
    }
    expected |= {(block, "source_only", 0, seed, None) for block, (seed, _) in enumerate(BLOCKS, 1)}
    actual = {
        (r["block"], r["method"], r["k"], r["training_seed"], r["support_seed"])
        for r in record["conditions"]
    }
    if (
        len(record["conditions"]) != 33
        or actual != expected
        or len(record["sources"]) != 3
        or {r["training_seed"] for r in record["sources"]} != {seed for seed, _ in BLOCKS}
    ):
        raise ResearchError("Identitas kondisi sumber/target tidak sesuai protokol.")
    locked = read_json(root / "sources-locked.json")
    if (
        locked["signature"] != sig
        or locked["target_updates"] != UPDATES
        or locked["target_lr"] != 5e-4
        or locked["ks"] != list(KS)
        or locked["blocks"] != manifest["blocks"]
    ):
        raise ResearchError("Aturan berubah setelah sumber dikunci.")
    lock_time = datetime.fromisoformat(locked["created_at"]).timestamp()
    data, mel = load_data(prepared, cache)
    source_split = split_indices(data["zoom_y"], data["zoom_order"], 42)
    source_split = {**source_split, "support": source_split["train"]}
    fresh_directory(output)
    verified, predictions, sources, artifact_hashes = [], [], {}, {}
    for row in record["sources"]:
        seed, folder = row["training_seed"], root / row["directory"]
        if locked["source_checkpoints"][str(seed)] != row["weights_sha256"]:
            raise ResearchError("Checkpoint sumber berbeda dari penguncian sebelum target.")
        initial = initialize_model(seed).state_dict()
        verified.append(
            replay(folder, row, initial, data, mel["zoom"], "zoom", source_split, sig, predictions)
        )
        sources[seed] = (
            folder,
            torch.load(folder / "best.pt", map_location="cpu", weights_only=True),
        )
        for path in folder.iterdir():
            if path.is_file():
                artifact_hashes[str(path.relative_to(root))] = file_hash(path)
        print(f"CPU source matches seed={seed}", flush=True)
    pairs = {}
    target_fit_ids = set()
    for row in record["conditions"]:
        seed, folder = row["training_seed"], root / row["directory"]
        if (folder / "config.json").stat().st_mtime < lock_time:
            raise ResearchError("Kondisi target dibuat sebelum penguncian sumber.")
        source_folder, source_state = sources[seed]
        if row["method"] == "source_only":
            cfg = read_json(folder / "config.json")
            test = np.flatnonzero(data["phone_order"] >= 15)
            if (
                cfg["signature"] != sig
                or cfg["test_ids"] != data["phone_ids"][test].tolist()
                or row["target_labels_used"] != 0
                or row["source_checkpoint_sha256"] != file_hash(source_folder / "best.pt")
            ):
                raise ResearchError("K=0 memakai anggaran/sumber/ID berbeda.")
            with np.load(source_folder / "normalization.npz", allow_pickle=False) as stats:
                mean, std = stats["mean"], stats["std"]
            scores = _score(
                initialize_model(seed, source_state),
                mel["phone"][test],
                torch.from_numpy(mean),
                torch.from_numpy(std),
                "cpu",
            )
            if measure(data["phone_y"][test], scores) != row["test"]:
                raise ResearchError("Replay K=0 berbeda.")
            with np.load(folder / "scores.npz", allow_pickle=False) as saved:
                if not (
                    np.array_equal(saved["indices"], test)
                    and np.array_equal(saved["y"], data["phone_y"][test])
                    and np.array_equal(saved["scores"].argmax(1), scores.argmax(1))
                    and np.allclose(saved["scores"], scores, rtol=1e-3, atol=1e-3)
                ):
                    raise ResearchError("Prediksi individual K=0 berbeda.")
                maximum_difference = float(np.abs(saved["scores"] - scores).max())
            for index, predicted in zip(test, scores.argmax(1), strict=True):
                predictions.append(
                    {
                        "domain": "phone",
                        "method": "source_only",
                        "training_seed": seed,
                        "support_seed": None,
                        "k": 0,
                        "event_id": str(data["phone_ids"][index]),
                        "actual": KEYS[data["phone_y"][index]],
                        "predicted": KEYS[predicted],
                        "actual_index": int(data["phone_y"][index]),
                        "predicted_index": int(predicted),
                    }
                )
            verified.append(
                {
                    "directory": row["directory"],
                    "method": "source_only",
                    "training_seed": seed,
                    "k": 0,
                    "confusion_match": True,
                    "max_absolute_score_difference": maximum_difference,
                }
            )
        else:
            split = calibration_split(
                data["phone_y"], data["phone_order"], row["k"], row["support_seed"]
            )
            initial = (
                source_state if row["method"] == "adapt" else initialize_model(seed).state_dict()
            )
            cfg = read_json(folder / "config.json")
            expected_source = (
                file_hash(source_folder / "best.pt") if row["method"] == "adapt" else None
            )
            target_fit_ids.update(cfg["support_ids"])
            if (
                cfg["source_checkpoint_sha256"] != expected_source
                or row["source_checkpoint_sha256"] != expected_source
            ):
                raise ResearchError("Sumber inisialisasi berbeda dari blok.")
            verified.append(
                replay(folder, row, initial, data, mel["phone"], "phone", split, sig, predictions)
            )
            pairs[row["block"], row["k"], row["method"]] = row
        for path in folder.iterdir():
            if path.is_file():
                artifact_hashes[str(path.relative_to(root))] = file_hash(path)
        print(f"CPU target matches {row['method']} seed={seed} K={row['k']}", flush=True)
    for block in range(1, 4):
        for k in KS:
            scratch, adapted = (pairs[block, k, method] for method in ("scratch", "adapt"))
            if scratch["batch_probes"] != adapted["batch_probes"]:
                raise ResearchError("Fingerprint pasangan berbeda.")
    with (output / "classifications.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    result = {
        "status": "passed",
        "created_at": now(),
        "results_sha256": file_hash(root / "results.json"),
        "signature": sig,
        "source_lock_sha256": file_hash(root / "sources-locked.json"),
        "source_train_labels_unique": 648,
        "source_validation_labels_unique": 72,
        "target_fit_labels_unique_in_campaign": len(target_fit_ids),
        "paired_batch_fingerprints_match": True,
        "new_fit_checkpoints_replayed": 33,
        "zero_shot_replayed": 3,
        "evaluations": verified,
        "artifact_hashes": artifact_hashes,
    }
    write_json(output / "audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/resnet-fewshot")
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/external/harrison-fewshot")
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "data/external/harrison-fewshot-features"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports/resnet-fewshot-audit")
    args = parser.parse_args()
    try:
        run(args.run, args.prepared, args.cache, args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
