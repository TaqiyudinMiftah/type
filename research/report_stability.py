"""Replay the activation controls and report every initialization."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ketik_audio.common import ResearchError, file_hash, fresh_directory, read_json, write_json
from ketik_audio.fewshot import _normalization, _score, calibration_split
from ketik_audio.harrison import KEYS, measure
from ketik_audio.stability import variant_model


def run(root: Path, prepared: Path, cache: Path, output: Path):
    torch.set_num_threads(4)
    record = read_json(root / "results.json")
    if record["status"] != "complete":
        raise ResearchError("Seluruh kontrol inisialisasi harus selesai.")
    if (
        file_hash(prepared / "segments.npz") != record["signature"]["segments_sha256"]
        or file_hash(cache / "features.npz") != record["signature"]["features_sha256"]
    ):
        raise ResearchError("Input pemeriksaan berubah.")
    expected = {(variant, seed) for variant in ["relu", "leaky"] for seed in [42, 43, 44]}
    if (
        len(record["conditions"]) != 6
        or {(row["variant"], row["training_seed"]) for row in record["conditions"]} != expected
    ):
        raise ResearchError("Enam kondisi yang ditetapkan belum lengkap.")
    with np.load(prepared / "segments.npz", allow_pickle=False) as archive:
        y, order, ids = archive["phone_y"], archive["phone_order"], archive["phone_ids"]
    with np.load(cache / "features.npz", allow_pickle=False) as archive:
        mel = archive["phone_mel"]
    split = calibration_split(y, order, 10, 101)
    support, test = split["support"], split["test"]
    mean, std = _normalization(mel, support)
    fresh_directory(output)
    audited, rows, histories, predictions = [], [], {}, []
    lookup = {}
    for row in sorted(record["conditions"], key=lambda r: (r["variant"], r["training_seed"])):
        variant, seed = row["variant"], row["training_seed"]
        directory = (
            Path(row["reused_directory"]) if "reused_directory" in row else root / row["directory"]
        )
        cfg = read_json(directory / "config.json")
        if (
            cfg["support_ids"] != ids[support].tolist()
            or cfg["test_ids"] != ids[test].tolist()
            or cfg["updates"] != 10_000
            or cfg["training_seed"] != seed
            or cfg["target_labels_used"] != 360
            or cfg["target_validation_labels"] != 0
        ):
            raise ResearchError("Kontrol memakai data/anggaran berbeda.")
        with np.load(directory / "normalization.npz", allow_pickle=False) as stats:
            if not (np.array_equal(mean, stats["mean"]) and np.array_equal(std, stats["std"])):
                raise ResearchError("Normalisasi bukan dukungan yang ditetapkan.")
        model, _ = variant_model(seed, variant)
        weights = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(weights)
        scores = _score(model, mel[test], torch.from_numpy(mean), torch.from_numpy(std), "cpu")
        measured = measure(y[test], scores)
        for field in ["confusion_matrix", "accuracy", "macro_f1", "top5", "correct", "n"]:
            if measured[field] != row["test"][field]:
                raise ResearchError(f"Replay berbeda: {variant}/{seed}/{field}")
        support_scores = _score(
            model, mel[support], torch.from_numpy(mean), torch.from_numpy(std), "cpu"
        )
        support_acc = float(np.mean(support_scores.argmax(axis=1) == y[support]))
        if support_acc != row["support_accuracy"]:
            raise ResearchError("Replay dukungan berbeda.")
        support_range = float(np.ptp(support_scores, axis=0).max())
        audited.append(
            {
                "variant": variant,
                "seed": seed,
                "confusion_match": True,
                "weights_sha256": file_hash(directory / "last.pt"),
                "reused_control": "reused_directory" in row,
            }
        )
        entry = {
            "variant": variant,
            "seed": seed,
            "support_accuracy": support_acc,
            "support_max_logit_range": support_range,
            "accuracy": measured["accuracy"],
            "macro_f1": measured["macro_f1"],
            "top5": measured["top5"],
            "correct": measured["correct"],
            "n": measured["n"],
            "training_seconds": row["training_seconds"],
        }
        rows.append(entry)
        lookup[variant, seed] = entry
        histories[variant, seed] = read_json(directory / "history.json")["updates"]
        for index, prediction in zip(test, scores.argmax(axis=1), strict=True):
            predictions.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "event_id": str(ids[index]),
                    "actual": KEYS[y[index]],
                    "predicted": KEYS[prediction],
                    "actual_index": int(y[index]),
                    "predicted_index": int(prediction),
                }
            )
        print(f"CPU replay matches: {variant} seed={seed}", flush=True)
    summary = []
    for variant in ["relu", "leaky"]:
        entries = [r for r in rows if r["variant"] == variant]
        summary.append(
            {
                "variant": variant,
                "accuracy_mean": float(np.mean([r["accuracy"] for r in entries])),
                "accuracy_sd": float(np.std([r["accuracy"] for r in entries], ddof=1)),
                "support_mean": float(np.mean([r["support_accuracy"] for r in entries])),
                "support_sd": float(np.std([r["support_accuracy"] for r in entries], ddof=1)),
                "constant_logit_runs": sum(r["support_max_logit_range"] == 0 for r in entries),
            }
        )
    differences = [
        lookup["leaky", seed]["accuracy"] - lookup["relu", seed]["accuracy"]
        for seed in [42, 43, 44]
    ]
    paired = {
        "mean": float(np.mean(differences)),
        "sd": float(np.std(differences, ddof=1)),
        "per_seed": dict(zip(map(str, [42, 43, 44]), differences, strict=True)),
    }
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for variant, color in [("relu", "#4287a8"), ("leaky", "#e18339")]:
        for seed in [42, 43, 44]:
            history = histories[variant, seed]
            supported = [r for r in history if "support_accuracy" in r]
            axes[0].plot(
                [r["step"] for r in supported],
                [r["support_accuracy"] * 100 for r in supported],
                marker="o",
                linewidth=1,
                color=color,
                linestyle={42: "-", 43: "--", 44: ":"}[seed],
                label=f"{variant}, seed {seed}",
            )
        entries = [lookup[variant, seed] for seed in [42, 43, 44]]
        offset = -0.16 if variant == "relu" else 0.16
        axes[1].bar(
            np.arange(3) + offset,
            [r["accuracy"] * 100 for r in entries],
            width=0.3,
            color=color,
            label=variant,
        )
    axes[0].set_xlabel("Update; dukungan sama, K=10")
    axes[0].set_ylabel("Akurasi dukungan tanpa augmentasi (%)")
    axes[0].legend(fontsize=7)
    axes[1].set_xticks(np.arange(3), [42, 43, 44])
    axes[1].set_xlabel("Seed inisialisasi")
    axes[1].set_ylabel("Top-1 uji (%)")
    axes[1].legend()
    for axis in axes:
        axis.set_ylim(0, 103)
    figure.suptitle(
        "ReLU asli vs LeakyReLU(0,01) pada gate mlp4\n"
        "Target ponsel, 360 label dukungan, 360 event uji tetap, 10.000 update",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(output / "kontrol.png", dpi=180, bbox_inches="tight")
    figure.savefig(output / "kontrol.pdf", bbox_inches="tight")
    plt.close(figure)
    lines = [
        "# Kontrol kestabilan aktivasi",
        "",
        "Target ponsel Harrison, K=10, seed dukungan 101, 360 label target dan 360 "
        "event uji yang sama. Lima fit baru dan satu kontrol ReLU seed 42 yang "
        "dipakai ulang dari diagnostik lama. Semua checkpoint akhir dievaluasi; "
        "tidak memilih seed atau checkpoint berdasarkan uji. Hasil tetap eksploratif.",
        "",
        "![Kontrol lintas seed](kontrol.png)",
        "",
        "## Semua hasil, termasuk kegagalan",
        "",
        "| Varian | Seed | Akurasi dukungan | Top-1 uji | Benar/uji | "
        "Macro-F1 | Rentang logit dukungan |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['seed']} | {row['support_accuracy'] * 100:.2f}% | "
            f"{row['accuracy'] * 100:.2f}% | {row['correct']}/{row['n']} | "
            f"{row['macro_f1']:.4f} | {row['support_max_logit_range']:.5g} |"
        )
    lines += [
        "",
        "| Varian | Rerata ± SD top-1 uji | Rerata ± SD dukungan | Run logit konstan |",
        "|---|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['variant']} | {row['accuracy_mean'] * 100:.2f}% ± "
            f"{row['accuracy_sd'] * 100:.2f} pp | "
            f"{row['support_mean'] * 100:.2f}% ± {row['support_sd'] * 100:.2f} pp | "
            f"{row['constant_logit_runs']}/3 |"
        )
    lines += [
        "",
        f"Selisih LeakyReLU − ReLU dipasangkan menurut seed: "
        f"{paired['mean'] * 100:.2f} ± {paired['sd'] * 100:.2f} poin persentase. "
        "Tiga inisialisasi tidak menetapkan keunggulan di populasi atau sesi baru. "
        "SD di sini variasi inisialisasi pada dukungan tetap, berbeda dari SD "
        "pilihan dukungan pada pilot kepala beku.",
        "",
        "## Apa yang diubah",
        "",
        "Hanya aktivasi `0.mlp4.1`: ReLU menjadi LeakyReLU(negative_slope=0.01). "
        "Bobot awal, data, batch, augmentasi, optimizer dan schedule sama untuk "
        "setiap pasangan seed. Kesetaraan semua tensor awal dan perbedaan satu "
        "modul diverifikasi dalam tes. Kode vendor dan checkpoint lama tidak "
        "diubah. Setiap model dilatih dari awal, Adam 5e-4, batch 16 dengan "
        "pengembalian, learning rate linear ke nol selama 10.000 update. Tidak "
        "ada label validasi target atau early stopping.",
        "",
        "Autopsi awal pada dukungan menunjukkan gate ReLU terakhir seluruhnya "
        "nol dan gradien input nol. Intervensi pada salinan memulihkan gradien; "
        "kontrol ini memeriksa apakah perubahan dari awal pelatihan membantu. "
        "Ini varian riset dari backbone kecil, bukan reproduksi literal model "
        "paper atau klaim SOTA. Keberhasilan satu gate tidak menjamin semua "
        "masalah optimisasi/generalisasi teratasi.",
        "",
        "## Verifikasi dan batas",
        "",
        "Enam kondisi diulang di CPU dari checkpoint tersimpan; confusion "
        "matrix, top-1/top-5, macro-F1 dan akurasi dukungan cocok persis. "
        "Identitas dukungan/uji dan normalisasi dihitung ulang untuk audit. "
        "Ekstraksi/pelatihan GPU hanya RX 6600 LE. Semua waktu dan gradien "
        "terukur tersimpan dalam history/result/summary JSON.",
        "",
        "Masih satu keyboard, dua kanal rekaman Harrison, dan pemisahan dalam "
        "WAV. Data uji sudah dipakai pada tahap sebelumnya; tidak ada holdout "
        "konfirmatori baru. Target ponsel dipilih karena kegagalan sebelumnya, "
        "sehingga ini diagnostik yang ditargetkan, bukan tolok ukur lintas domain "
        "menyeluruh. Ketidakpastian phone/J tetap berlaku. K=10 saja tidak "
        "menetapkan jumlah minimal contoh atau efisiensi label adaptasi.",
        "",
        "Protokol: `research/STABILITY_PROTOCOL.md`. "
        f"SHA256 hasil: `{file_hash(root / 'results.json')}`.",
        "",
    ]
    (output / "laporan.md").write_text("\n".join(lines), encoding="utf-8")
    for filename, entries in [("per-run.csv", rows), ("classifications.csv", predictions)]:
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
            writer.writeheader()
            writer.writerows(entries)
    write_json(
        output / "summary.json",
        {
            "source_results_sha256": file_hash(root / "results.json"),
            "runs": rows,
            "groups": summary,
            "paired_difference": paired,
            "cpu_replay": audited,
        },
    )
    print(f"Laporan dan audit enam kontrol selesai: {output / 'laporan.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.run, args.prepared, args.cache, args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
