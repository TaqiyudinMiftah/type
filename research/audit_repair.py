"""Replay every repair checkpoint on CPU and verify the source-selected decision."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from ketik_audio import cnn_repair, repair, stem_repair
from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, read_json, write_json
from ketik_audio.fewshot import ROOT, _normalization, _score, calibration_split
from ketik_audio.harrison import KEYS, measure
from ketik_audio.repair import (
    SEEDS,
    development_split,
    load_inputs,
    qualifies,
)

OLD_HASHES = {
    "runs/harrison-reproduction/results.json": (
        "f23fb66f1fe422fa9ef99c68f6970e9d7e15089eee577d2bdc92bb3c31171433"
    ),
    "runs/fewshot-pilot/results.json": (
        "0b033014568b62ce8b9ed8f1773ed7d33af50c23740d2d99e1763e671416cba1"
    ),
    "runs/fewshot-optimization-diagnostic/results.json": (
        "eb52e50c09f47eba61e80f761164a0fd86ddd668105d0fa737b0186fe01b2b88"
    ),
    "runs/stability-gate/results.json": (
        "0054c507f7664e3e54e03aefe5d3547af9ececfe2bbd52317d7fb1ef8a298511"
    ),
    "src/ketik_audio/stability.py": (
        "e0d60ba7dc0933c58885c42bf02795c4a19273d1c7212d3a474f07ed0c2fcb97"
    ),
}


def run(development, confirmation, prepared, cache, output):
    torch.set_num_threads(4)
    dev, target = (read_json(folder / "results.json") for folder in (development, confirmation))
    decision = read_json(development / "decision.json")
    chosen = decision["variant"]
    runner = {"compact-resnet": cnn_repair, "wide-stem-residual": stem_repair}.get(chosen, repair)
    sig = runner.signature(prepared, cache)
    earlier_records = []
    earlier_stages = []
    if runner in (stem_repair, cnn_repair):
        earlier_stages.append((repair, stem_repair.PARENT))
    if runner is cnn_repair:
        earlier_stages.append((stem_repair, cnn_repair.PARENT))
    for previous_runner, directory in earlier_stages:
        earlier = read_json(directory / "results.json")
        if (
            earlier["status"] != "complete"
            or earlier["selected_variant"] is not None
            or len(earlier["conditions"]) != 3 * len(previous_runner.CANDIDATES)
            or earlier["signature"] != previous_runner.signature(prepared, cache)
        ):
            raise ResearchError("Kandidat terdahulu harus lengkap dan gagal seleksi pengembangan.")
        for variant in previous_runner.CANDIDATES:
            previous_rows = [r for r in earlier["conditions"] if r["variant"] == variant]
            if len(previous_rows) != 3 or qualifies(previous_rows):
                raise ResearchError("Ada kandidat terdahulu yang seharusnya dipilih.")
        earlier_records.append((directory, earlier, "zoom", "development"))
    if dev["status"] != "complete" or target["status"] != "complete":
        raise ResearchError("Fit pengembangan/konfirmasi belum lengkap.")
    if dev["signature"] != sig or target["signature"] != sig or decision["signature"] != sig:
        raise ResearchError("Kode/data/protokol berubah sejak pemilihan kandidat.")
    if decision["development_results_sha256"] != file_hash(development / "results.json"):
        raise ResearchError("Hasil pengembangan berbeda dari keputusan terkunci.")
    if target["decision_sha256"] != file_hash(development / "decision.json"):
        raise ResearchError("Keputusan berbeda dari yang dipakai konfirmasi.")
    if decision != read_json(confirmation / "decision.json"):
        raise ResearchError("Salinan keputusan berubah.")
    target_manifest = read_json(confirmation / "manifest.json")
    if decision["created_at"] >= target_manifest["created_at"]:
        raise ResearchError("Keputusan tidak mendahului konfirmasi ponsel.")
    if decision["selection_domain"] != "zoom" or decision["phone_labels_used_for_selection"] != 0:
        raise ResearchError("Pemilihan kandidat memakai label ponsel.")
    candidates_run = []
    for variant in runner.CANDIDATES:
        rows = [r for r in dev["conditions"] if r["variant"] == variant]
        if not rows:
            break
        if len(rows) != 3 or {r["training_seed"] for r in rows} != set(SEEDS):
            raise ResearchError("Kondisi pengembangan tidak lengkap/duplikat.")
        candidates_run.append(variant)
        if qualifies(rows):
            if variant != chosen:
                raise ResearchError("Kandidat yang dipilih bukan kandidat pertama yang lulus.")
            break
    if not candidates_run or candidates_run[-1] != chosen:
        raise ResearchError("Kandidat terpilih tidak lulus pengembangan.")
    if len(dev["conditions"]) != 3 * len(candidates_run):
        raise ResearchError("Ada fit tambahan di luar urutan seleksi.")
    if len(target["conditions"]) != 3 or {
        (r["variant"], r["training_seed"]) for r in target["conditions"]
    } != {(chosen, seed) for seed in SEEDS}:
        raise ResearchError("Konfirmasi harus memuat ketiga seed, tanpa pemilihan.")
    for filename, expected in OLD_HASHES.items():
        if file_hash(ROOT / filename) != expected:
            raise ResearchError(f"Eksperimen lama berubah: {filename}")
    data, features = load_inputs(prepared, cache)
    fresh_directory(output)
    verified, predictions, artifacts = [], [], {}
    records = [
        (development, dev, "zoom", "development"),
        (confirmation, target, "phone", "confirmation"),
    ]
    records = earlier_records + records
    for root, record, domain, stage in records:
        y, order, ids, mel = (
            data[f"{domain}_y"],
            data[f"{domain}_order"],
            data[f"{domain}_ids"],
            features[f"{domain}_mel"],
        )
        split = (
            development_split(y, order)
            if stage == "development"
            else calibration_split(y, order, 10, 101)
        )
        eval_name = "validation" if stage == "development" else "test"
        support, evaluation = split["support"], split[eval_name]
        mean, std = _normalization(mel, support)
        for row in record["conditions"]:
            folder = root / row["directory"]
            cfg = read_json(folder / "config.json")
            if (
                cfg["support_ids"] != ids[support].tolist()
                or cfg[f"{eval_name}_ids"] != ids[evaluation].tolist()
            ):
                raise ResearchError("Identitas data berbeda dari split yang ditetapkan.")
            if (
                cfg["domain"] != domain
                or cfg["stage"] != stage
                or cfg["training_seed"] != row["training_seed"]
                or cfg["variant"] != row["variant"]
                or cfg["updates"] != (6000 if stage == "development" else 10_000)
                or cfg["lr"] != 5e-4
                or cfg["batch_size"] != 16
            ):
                raise ResearchError("Konfigurasi/anggaran tidak sesuai.")
            with np.load(folder / "normalization.npz", allow_pickle=False) as stats:
                if not np.array_equal(stats["mean"], mean) or not np.array_equal(stats["std"], std):
                    raise ResearchError("Normalisasi memakai contoh di luar dukungan.")
            if file_hash(folder / "last.pt") != row["weights_sha256"]:
                raise ResearchError("Checkpoint berubah.")
            model, _ = runner.repaired_model(row["training_seed"], row["variant"])
            model.load_state_dict(
                torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
            )
            scores = _score(
                model, mel[evaluation], torch.from_numpy(mean), torch.from_numpy(std), "cpu"
            )
            support_scores = _score(
                model, mel[support], torch.from_numpy(mean), torch.from_numpy(std), "cpu"
            )
            metrics = measure(y[evaluation], scores)
            if (
                metrics != row[eval_name]
                or float((support_scores.argmax(1) == y[support]).mean()) != row["support_accuracy"]
            ):
                raise ResearchError(f"Metrik replay CPU berbeda: {folder}")
            with np.load(folder / "scores.npz", allow_pickle=False) as saved:
                if (
                    not np.array_equal(saved["indices"], evaluation)
                    or not np.array_equal(saved["support"], support)
                    or not np.array_equal(saved["y"], y[evaluation])
                ):
                    raise ResearchError("Indeks/label skor tersimpan berbeda.")
                if not np.array_equal(
                    saved["scores"].argmax(1), scores.argmax(1)
                ) or not np.array_equal(
                    saved["support_scores"].argmax(1), support_scores.argmax(1)
                ):
                    raise ResearchError("Prediksi individual replay CPU berbeda.")
                if not np.allclose(saved["scores"], scores, rtol=1e-3, atol=1e-3):
                    raise ResearchError("Skor CPU/GPU di luar toleransi.")
                max_diff = float(np.max(np.abs(saved["scores"] - scores)))
            history = read_json(folder / "history.json")["updates"]
            if history[-1]["step"] != cfg["updates"]:
                raise ResearchError("Anggaran update tidak lengkap.")
            if stage == "confirmation" and any(
                "validation_accuracy" in h or "test_accuracy" in h for h in history
            ):
                raise ResearchError("Uji target digunakan selama fit.")
            verified.append(
                {
                    "stage": stage,
                    "variant": row["variant"],
                    "seed": row["training_seed"],
                    "confusion_match": True,
                    "support_predictions_match": True,
                    "budget_and_normalization_match": True,
                    "max_absolute_score_difference": max_diff,
                }
            )
            for index, pred in zip(evaluation, scores.argmax(1), strict=True):
                predictions.append(
                    {
                        "stage": stage,
                        "variant": row["variant"],
                        "seed": row["training_seed"],
                        "event_id": str(ids[index]),
                        "actual": KEYS[y[index]],
                        "predicted": KEYS[pred],
                        "actual_index": int(y[index]),
                        "predicted_index": int(pred),
                    }
                )
            for path in folder.iterdir():
                if path.is_file():
                    artifacts[str(path.relative_to(ROOT))] = file_hash(path)
            print(
                f"CPU replay matches {stage} {row['variant']} seed={row['training_seed']}",
                flush=True,
            )
    with (output / "classifications.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    result = {
        "status": "passed",
        "created_at": now(),
        "signature": sig,
        "decision_sha256": file_hash(development / "decision.json"),
        "development_results_sha256": file_hash(development / "results.json"),
        "confirmation_results_sha256": file_hash(confirmation / "results.json"),
        "decision_precedes_target_fit": True,
        "old_artifacts_unchanged": OLD_HASHES,
        "checkpoints": verified,
        "artifact_hashes": artifacts,
    }
    write_json(output / "audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, default=ROOT / "runs/cnn-repair-development")
    parser.add_argument("--confirmation", type=Path, default=ROOT / "runs/cnn-repair-confirmation")
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/external/harrison-fewshot")
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "data/external/harrison-fewshot-features"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports/repair-audit")
    args = parser.parse_args()
    try:
        run(args.development, args.confirmation, args.prepared, args.cache, args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
