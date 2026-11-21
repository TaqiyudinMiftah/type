"""Standalone Indonesian Markdown report and exportable figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import LABELS, ResearchError, file_hash, fresh_directory, now, read_json, write_json


def report(models: list[Path], output: Path, evaluations: list[Path] | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if not models or len(set(path.resolve() for path in models)) != len(models):
        raise ResearchError("Berikan minimal satu direktori model yang unik.")
    if evaluations is not None and len(evaluations) != len(models):
        raise ResearchError("Jumlah --evaluations harus sama dengan --models dan urutannya sama.")
    records = []
    for i, model in enumerate(models):
        metadata = read_json(model / "model.json")
        training = read_json(model / "training.json")
        evaluation_dir = evaluations[i] if evaluations else model / "evaluation"
        evaluation_path = evaluation_dir / "evaluation.json"
        evaluation = read_json(evaluation_path) if evaluation_path.exists() else None
        if evaluation and evaluation["model_sha256"] != file_hash(model / "model.json"):
            raise ResearchError(f"Evaluasi tidak cocok dengan model: {model}")
        records.append((metadata, training, evaluation))
    fingerprints = {row[0]["source_manifest"]["features_sha256"] for row in records}
    if len(fingerprints) != 1:
        raise ResearchError("Perbandingan model harus menggunakan dataset/split yang sama.")
    test_hashes = {json.dumps(row[2]["test_hashes"], sort_keys=True) for row in records if row[2]}
    if len(test_hashes) > 1:
        raise ResearchError("Perbandingan model harus menggunakan rekaman uji yang sama.")
    cnn_settings = {
        (
            metadata["batch_size"],
            metadata["max_epochs"],
            metadata["patience"],
            metadata["learning_rate"],
            tuple(sorted(row["seed"] for row in training.get("seeds", []))),
            metadata["environment"]["packages"]["torch"].split("+")[0],
        )
        for metadata, training, _ in records
        if metadata["kind"] == "cnn"
    }
    if len(cnn_settings) > 1:
        raise ResearchError(
            "Perbandingan CPU/GPU memerlukan versi dasar PyTorch, seed, "
            "batch, learning rate, patience, dan batas epoch yang sama."
        )
    fresh_directory(output)
    synthetic = records[0][0]["synthetic"]
    source = records[0][0]["source_manifest"]
    heading = "UJI PERANGKAT LUNAK — DATA SINTETIS" if synthetic else "HASIL REKAMAN KEYBOARD"
    lines = [
        "# Laporan Riset Pengenalan Huruf dari Suara Ketikan",
        "",
        f"**{heading}**",
        "",
        f"Dibuat: {now()}",
        "",
        "## Tujuan dan metode",
        "",
        "Membandingkan MFCC–SVM dengan CNN kecil untuk mengenali 26 tombol a–z "
        "pada satu keyboard dan pengguna, dengan pengujian lintas sesi.",
        "",
        "Sesi 1–4 digunakan untuk pelatihan, sesi 5 untuk pemilihan konfigurasi, "
        "dan sesi 6 untuk pengujian akhir. Statistik normalisasi hanya dipelajari "
        "dari sesi 1–4. Ambang deteksi dipilih menggunakan sesi 5.",
        "",
        "Rangkaian referensi mencakup huruf aktual saat salah tekan. Interval kontrol "
        "istirahat dikeluarkan berdasarkan waktu protokol. Potongan klasifikasi yang "
        "tumpang tindih dengan kontrol atau ketukan lain ditolak dengan kebijakan "
        "yang sama pada latihan dan pengujian.",
        "",
        f"Dataset: {source['counts']['train']} contoh latih dan "
        f"{source['counts']['validation']} contoh validasi; "
        f"{len(source['rejected_events'])} potongan ditolak saat persiapan.",
        "",
        f"Fingerprint fitur: `{source['features_sha256']}`.",
        "",
    ]
    if synthetic:
        lines.extend(
            [
                "Data sintetis hanya memverifikasi alur perangkat lunak. Angka di bawah "
                "tidak mengukur kemampuan mengenali suara keyboard nyata.",
                "",
            ]
        )
    lines.extend(
        [
            "## Hasil",
            "",
            "Akurasi dan macro-F1 diukur pada potongan dengan penanda tombol yang "
            "diketahui. F1 deteksi dan CER mengukur alur audio saja; CER mencakup "
            "ketukan hilang, tambahan, dan salah klasifikasi.",
            "",
            "| Model / pelatihan | Akurasi | Macro-F1 | F1 deteksi | CER | Waktu latih |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for metadata, training, evaluation in records:
        name = f"{metadata['kind'].upper()} / {metadata['device']['selected']}"
        seconds = training["training_seconds"]
        if not evaluation:
            lines.append(f"| {name} | menunggu uji | — | — | — | {seconds:.2f} dtk |")
            continue
        measured = evaluation["classification"]
        stats = evaluation.get("seed_statistics")
        accuracy = (
            f"{stats['accuracy']['mean']:.3f} ± {stats['accuracy']['std']:.3f}"
            if stats
            else f"{measured['accuracy']:.3f}"
        )
        f1 = (
            f"{stats['macro_f1']['mean']:.3f} ± {stats['macro_f1']['std']:.3f}"
            if stats
            else f"{measured['macro_f1']:.3f}"
        )
        cer = (
            f"{stats['cer']['mean']:.3f} ± {stats['cer']['std']:.3f}"
            if stats
            else f"{evaluation['end_to_end']['cer']:.3f}"
        )
        lines.append(
            f"| {name} | {accuracy} | {f1} | {evaluation['detection']['f1']:.3f} "
            f"| {cer} | {seconds:.2f} dtk |"
        )
    lines.extend(
        [
            "",
            "CNN dengan beberapa seed dilaporkan sebagai rata-rata ± simpangan baku "
            "populasi. Waktu latih CNN adalah total seluruh seed; waktu SVM mencakup "
            "pencarian enam kombinasi parameter. Keduanya mencakup prosedur pelatihan berbeda.",
            "",
        ]
    )
    for index, (metadata, training, evaluation) in enumerate(records):
        name = f"{metadata['kind'].upper()} / {metadata['device']['selected']}"
        lines.extend([f"### {name}", ""])
        if metadata["device"].get("fallback_reason"):
            lines.extend([f"Backend: {metadata['device']['fallback_reason']}", ""])
        if not evaluation:
            lines.extend(
                ["Sesi uji belum dievaluasi. Kesimpulan akurasi akhir belum tersedia.", ""]
            )
            continue
        if metadata.get("seed") is not None:
            lines.extend(
                [
                    f"Seed model pilihan: {metadata['seed']}; dipilih dari validasi. "
                    "Confusion matrix berikut memakai model pilihan tersebut.",
                    "",
                ]
            )
        matrix = np.asarray(evaluation["classification"]["confusion_matrix"])
        fig, axis = plt.subplots(figsize=(10, 9))
        picture = axis.imshow(matrix, cmap="Blues")
        qualifier = " — DATA SINTETIS" if synthetic else ""
        axis.set(
            xticks=range(26),
            yticks=range(26),
            xticklabels=LABELS,
            yticklabels=LABELS,
            xlabel="Prediksi",
            ylabel="Label sebenarnya",
            title=f"Confusion matrix — {name}{qualifier}",
        )
        colorbar = fig.colorbar(picture, ax=axis)
        colorbar.locator = MaxNLocator(integer=True)
        colorbar.update_ticks()
        fig.tight_layout()
        image_name = f"model-{index}-confusion.png"
        fig.savefig(output / image_name, dpi=160)
        plt.close(fig)
        lines.extend([f"![Confusion matrix {name}]({image_name})", ""])
        wrong = matrix.copy()
        np.fill_diagonal(wrong, 0)
        pairs = sorted(
            (
                (int(wrong[i, j]), LABELS[i], LABELS[j])
                for i in range(26)
                for j in range(26)
                if wrong[i, j]
            ),
            reverse=True,
        )[:5]
        lines.extend(
            [
                "Kesalahan dominan: "
                + (
                    "; ".join(
                        f"{actual} → {predicted}: {count}" for count, actual, predicted in pairs
                    )
                    if pairs
                    else "tidak ada pada sampel uji ini"
                )
                + ".",
                "",
            ]
        )
        d = evaluation["detection"]
        coverage = evaluation["classification_coverage"]
        lines.extend(
            [
                f"Cakupan klasifikasi: {coverage['evaluated']} dari "
                f"{coverage['accepted_markers']} penanda; {coverage['excluded']} potongan "
                "dikeluarkan karena batas audio atau ketukan/kontrol yang tumpang tindih.",
                "",
                f"Deteksi: {d['matched']} cocok, {d['missed']} hilang, dan "
                f"{d['extra']} tambahan; toleransi {d['tolerance_ms']:.0f} ms.",
                "",
                f"Inferensi batch: {evaluation['timing']['ms_per_event_batched']:.3f} "
                "ms/ketukan setelah pemanasan, di luar pemrosesan audio. "
                "Ini ukuran throughput batch, bukan latensi aplikasi langsung.",
                "",
            ]
        )
        if not synthetic:
            lines.extend(
                [
                    "Target eksploratif macro-F1 ≥0,70 untuk model pilihan: "
                    + ("tercapai." if evaluation["target_met"] else "belum tercapai."),
                    "",
                ]
            )
        if training.get("seeds"):
            actual_epochs = "; ".join(
                f"seed {seed['seed']}: {len(seed['history'])} epoch" for seed in training["seeds"]
            )
            lines.extend(
                [
                    f"Epoch aktual: {actual_epochs}. Batch {metadata['batch_size']}, "
                    f"batas {metadata['max_epochs']} epoch. Perbedaan early stopping "
                    "perlu diperhitungkan saat membandingkan waktu perangkat.",
                    "",
                ]
            )
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            fig.suptitle(f"Pelatihan {name}{qualifier}")
            for seed in training["seeds"]:
                history = seed["history"]
                epochs = [row["epoch"] for row in history]
                axes[0].plot(
                    epochs, [row["train_loss"] for row in history], label=f"seed {seed['seed']}"
                )
                axes[1].plot(
                    epochs,
                    [row["validation_macro_f1"] for row in history],
                    label=f"seed {seed['seed']}",
                )
            axes[0].set(xlabel="Epoch", ylabel="Loss latih")
            axes[1].set(xlabel="Epoch", ylabel="Macro-F1 validasi")
            for axis in axes:
                axis.legend()
            fig.tight_layout()
            curve_name = f"model-{index}-training.png"
            fig.savefig(output / curve_name, dpi=160)
            plt.close(fig)
            lines.extend([f"![Kurva pelatihan]({curve_name})", ""])
    lines.extend(
        [
            "## Keterbatasan dan tindak lanjut",
            "",
            "Eksperimen ini memakai ketukan terpisah dengan jeda terpandu pada satu "
            "keyboard. Hasil belum mengukur pengetikan kalimat alami, perpindahan "
            "pengguna/keyboard, ataupun perubahan kebisingan. Skor prediksi belum "
            "dikalibrasi sebagai probabilitas. CER dapat melebihi 1 ketika banyak "
            "ketukan tambahan terdeteksi.",
            "",
            "Perbandingan CPU/GPU menggunakan dataset, seed, batch, dan batas epoch "
            "yang sama. Jika backend GPU belum lolos tes forward/backward, "
            "laporkan hasil CPU dan kendala kompatibilitas tanpa menyimpulkan "
            "performa GPU.",
            "",
            "Jika target belum tercapai, tinjau pasangan huruf yang tertukar dan "
            "kualitas sinkronisasi. Perubahan metode memerlukan eksperimen baru "
            "dengan sesi uji baru; hindari menyetel model pada sesi 6 yang telah dibuka.",
            "",
            "## Rujukan",
            "",
            "- [Harrison dkk. (2023)](https://arxiv.org/abs/2308.01074): "
            "pengenalan ketukan menggunakan mel-spectrogram dan deep learning.",
            "- [Taheritajar & Rahaeimehr (2024)](https://arxiv.org/abs/2403.08740): "
            "pengaruh pola mengetik dan kondisi nyata.",
            "",
            "Angka akurasi antarpenelitian perlu dibaca bersama perangkat, dataset, "
            "protokol pemisahan data, dan metrik masing-masing.",
            "",
        ]
    )
    (output / "laporan.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(
        output / "results.json",
        {
            "synthetic": synthetic,
            "models": [
                {"model": metadata, "training": training, "evaluation": evaluation}
                for metadata, training, evaluation in records
            ],
        },
    )
    return output / "laporan.md"
