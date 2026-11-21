"""Replay and report the separately specified scratch optimization diagnostic."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ketik_audio.common import ResearchError, file_hash, fresh_directory, read_json, write_json
from ketik_audio.fewshot import _normalization, _score, calibration_split
from ketik_audio.harrison import _reference_model, measure


def report(primary: Path, diagnostic: Path, prepared: Path, cache: Path, output: Path):
    import torch

    torch.set_num_threads(4)
    original = read_json(primary / "results.json")
    record = read_json(diagnostic / "results.json")
    if record["status"] != "complete" or original["status"] != "complete":
        raise ResearchError("Pilot/diagnostik belum selesai.")
    if record["signature"]["primary_results_sha256"] != file_hash(primary / "results.json"):
        raise ResearchError("Hasil primer berubah.")
    if file_hash(prepared / "segments.npz") != original["signature"]["segments_sha256"]:
        raise ResearchError("Potongan berbeda.")
    if file_hash(cache / "features.npz") != original["signature"]["features_sha256"]:
        raise ResearchError("Fitur berbeda.")
    data = np.load(prepared / "segments.npz", allow_pickle=False)
    features = np.load(cache / "features.npz", allow_pickle=False)
    fresh_directory(output)
    lines = [
        "# Diagnostik optimisasi scratch",
        "",
        "Tambahan pascahasil, dipicu oleh rendahnya akurasi dukungan scratch pada "
        "1000 update. Tidak mengganti protokol atau hasil utama. Dua fit tambahan "
        "memakai K=10, seed dukungan 101, 360 label target dan 360 event uji tetap "
        "per target. Seluruh pembanding di bawah memakai dukungan seed 101 yang sama; "
        "bukan rerata tiga draw.",
        "",
        "| Target | Metode | Update | Top-1 uji | Benar/uji | Akurasi dukungan | Waktu fit |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    audit_rows, comparison = [], []
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, row in zip(axes, record["conditions"], strict=True):
        target = row["target"]
        out = diagnostic / target
        config = read_json(out / "config.json")
        y, order = data[f"{target}_y"], data[f"{target}_order"]
        split = calibration_split(y, order, 10, 101)
        support, test = split["support"], split["test"]
        if (
            config["support_ids"] != data[f"{target}_ids"][support].tolist()
            or config["test_ids"] != data[f"{target}_ids"][test].tolist()
            or config["target_labels_used"] != 360
            or config["target_validation_labels"] != 0
        ):
            raise ResearchError("Anggaran/identitas diagnostik berbeda.")
        mean, std = _normalization(features[f"{target}_mel"], support)
        stats = np.load(out / "normalization.npz", allow_pickle=False)
        if not (np.array_equal(mean, stats["mean"]) and np.array_equal(std, stats["std"])):
            raise ResearchError("Normalisasi diagnostik berbeda.")
        model, _ = _reference_model()
        model.load_state_dict(torch.load(out / "last.pt", map_location="cpu", weights_only=True))
        scores = _score(
            model,
            features[f"{target}_mel"][test],
            torch.from_numpy(mean),
            torch.from_numpy(std),
            "cpu",
        )
        measured = measure(y[test], scores)
        for field in ["confusion_matrix", "accuracy", "macro_f1", "top5", "correct", "n"]:
            if measured[field] != row["test"][field]:
                raise ResearchError(f"Replay CPU diagnostik berbeda: {target}/{field}")
        audit_rows.append(
            {
                "target": target,
                "confusion_match": True,
                "weights_sha256": file_hash(out / "last.pt"),
            }
        )
        entries = [
            (r["method"], 1000, r)
            for r in original["conditions"]
            if r["target"] == target
            and r["k"] == 10
            and r["support_seed"] == 101
            and r["method"] in ["coatnet_scratch", "coatnet_adapt"]
        ]
        entries.append(("coatnet_scratch_long", 10_000, row))
        for method, updates, r in entries:
            lines.append(
                f"| {target} | {method} | {updates} | {r['test']['accuracy'] * 100:.2f}% | "
                f"{r['test']['correct']}/{r['test']['n']} | {r['support_accuracy'] * 100:.2f}% | "
                f"{r['training_seconds']:.2f} s |"
            )
            comparison.append(
                {
                    "target": target,
                    "method": method,
                    "updates": updates,
                    "accuracy": r["test"]["accuracy"],
                    "macro_f1": r["test"]["macro_f1"],
                    "support_accuracy": r["support_accuracy"],
                    "training_seconds": r["training_seconds"],
                }
            )
        history = read_json(out / "history.json")["updates"]
        supported = [r for r in history if "support_accuracy" in r]
        axis.plot(
            [r["step"] for r in supported],
            [r["support_accuracy"] * 100 for r in supported],
            marker="o",
            color="#4c78a8",
        )
        axis.set_title(target)
        axis.set_xlabel("Update scratch; schedule linear selama 10.000 update")
        axis.set_ylim(0, 103)
    axes[0].set_ylabel("Akurasi dukungan tanpa augmentasi (%)")
    figure.suptitle("Diagnostik optimisasi; evaluasi uji hanya di update terakhir")
    figure.tight_layout()
    figure.savefig(output / "dukungan.png", dpi=180, bbox_inches="tight")
    figure.savefig(output / "dukungan.pdf", bbox_inches="tight")
    plt.close(figure)
    lines += [
        "",
        "![Akurasi dukungan](dukungan.png)",
        "",
        "Scratch panjang dilatih ulang dari awal dengan Adam lr awal 5e-4 dan "
        "schedule linear ke nol selama 10.000 update. Kurva dukungan tidak sama "
        "dengan run utama yang schedule-nya berakhir di 1000 update. Tidak ada "
        "early stopping atau pemilihan checkpoint dari hasil uji. Adaptasi tetap "
        "1000 update dengan lr awal 1e-4 dan membawa biaya pretraining sumber.",
        "",
        "Gunakan hasil ini untuk menilai keterbatasan optimisasi baseline, bukan "
        "mengeklaim keunggulan arsitektur atau efisiensi label murni. Satu draw per "
        "target tidak cukup mengukur variasi atau menetapkan minimum jumlah label; "
        "uji tetap dalam rekaman yang sama. Protokol diagnostik dicatat setelah "
        "hasil K=1 pertama dan sebelum kondisi K=10 selesai, dalam "
        "`research/FEWSHOT_DIAGNOSTIC_PROTOCOL.md` dan snapshot pemicu.",
        "",
        "Replay CPU dua model menghasilkan confusion matrix dan metrik yang "
        "sama persis dengan run GPU. Split dan statistik dukungan juga diverifikasi.",
        "",
        f"Hasil primer SHA256: `{file_hash(primary / 'results.json')}`. "
        f"Hasil diagnostik SHA256: `{file_hash(diagnostic / 'results.json')}`.",
        "",
    ]
    write_json(
        output / "summary.json",
        {
            "comparison": comparison,
            "cpu_replay": audit_rows,
            "source_results_sha256": file_hash(diagnostic / "results.json"),
        },
    )
    (output / "laporan.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Diagnostik dan replay CPU selesai: {output / 'laporan.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report(args.primary, args.diagnostic, args.prepared, args.cache, args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
