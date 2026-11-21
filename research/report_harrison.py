"""Render measured external-data experiments as a standalone research report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.patches import FancyArrowPatch, Rectangle

from ketik_audio.common import ResearchError, file_hash, fresh_directory, read_json, write_json
from ketik_audio.harrison import KEYS


def percentage(value):
    return f"{100 * value:.2f}%"


def save_figure(figure, output, name):
    figure.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    figure.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def figures(root, output, results):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, domain, published in zip(axes, ["phone", "zoom"], [95, 93], strict=True):
        baseline = [
            r["test"]["accuracy"] * 100
            for r in results
            if r["kind"] == "mfcc-svm-control" and r["split"] == "random" and r["domain"] == domain
        ]
        neural = next(
            r for r in results if r["domain"] == domain and r["kind"].startswith("coatnet")
        )
        axis.bar(
            [0, 1],
            [np.mean(baseline), neural["test"]["accuracy"] * 100],
            color=["#4287a8", "#e18339"],
        )
        axis.errorbar(
            0,
            np.mean(baseline),
            yerr=np.std(baseline, ddof=1),
            fmt="none",
            capsize=5,
            color="black",
        )
        axis.axhline(published, color="#535353", linestyle="--", label=f"Paper: {published}%")
        axis.axhline(100 / 36, color="#999999", linestyle=":", label="Peluang acak 1/36")
        axis.set_xticks([0, 1], ["SVM: 3 split", "CoAtNet: seed 42"])
        axis.set_title("Ponsel" if domain == "phone" else "Zoom")
        axis.set_ylim(0, 105)
        axis.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("Top-1 uji (%)")
    figure.suptitle(
        "Harrison 2023 — percobaan reproduksi independen\n"
        "SVM: rerata ± SD tiga split; CoAtNet: satu inisialisasi",
        fontsize=12,
    )
    figure.tight_layout()
    save_figure(figure, output, "perbandingan")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    for column, domain in enumerate(["phone", "zoom"]):
        row = next(r for r in results if r["domain"] == domain and r["kind"].startswith("coatnet"))
        history = read_json(root / row["directory"] / "history.json")["epochs"]
        axes[0, column].plot(
            [h["epoch"] for h in history], [h["loss"] for h in history], color="#4287a8"
        )
        valid = [h for h in history if "validation_accuracy" in h]
        axes[1, column].plot(
            [h["epoch"] for h in valid],
            [h["validation_accuracy"] * 100 for h in valid],
            color="#e18339",
        )
        axes[1, column].axvline(
            row["best_epoch"],
            linestyle="--",
            color="#555555",
            label=f"Checkpoint epoch {row['best_epoch']}",
        )
        axes[0, column].set_title("Ponsel" if domain == "phone" else "Zoom")
        axes[1, column].set_xlabel("Epoch")
        axes[1, column].set_ylim(0, 101)
        axes[1, column].legend(fontsize=8)
    axes[0, 0].set_ylabel("Cross-entropy latihan")
    axes[1, 0].set_ylabel("Top-1 validasi (%)")
    figure.suptitle("Kurva CoAtNet; checkpoint dipilih hanya menggunakan validasi")
    figure.tight_layout()
    save_figure(figure, output, "kurva")

    figure, axes = plt.subplots(2, 2, figsize=(15, 13))
    for row_number, kind in enumerate(
        ["mfcc-svm-control", "coatnet-reference-reproduction-attempt"]
    ):
        for column, domain in enumerate(["phone", "zoom"]):
            row = next(
                r
                for r in results
                if r["kind"] == kind
                and r["domain"] == domain
                and r["split"] == "random"
                and r["seed"] == 42
            )
            matrix = np.asarray(row["test"]["confusion_matrix"])
            normalized = matrix / matrix.sum(axis=1, keepdims=True)
            axis = axes[row_number, column]
            rendered = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
            axis.set_xticks(range(36), list(KEYS), fontsize=7)
            axis.set_yticks(range(36), list(KEYS), fontsize=7)
            axis.set_xlabel("Prediksi")
            axis.set_ylabel("Label sumber")
            axis.set_title(
                f"{'SVM' if row_number == 0 else 'CoAtNet'} / {domain}: "
                f"{percentage(row['test']['accuracy'])} (n=180)"
            )
    figure.colorbar(rendered, ax=axes.ravel().tolist(), shrink=0.6, label="Proporsi per kelas")
    figure.suptitle("Confusion matrix uji, split acak seed 42 — lima contoh per kelas")
    figure.subplots_adjust(top=0.94, hspace=0.23, wspace=0.2, right=0.88)
    save_figure(figure, output, "confusion")

    figure, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for axis, domain in zip(axes, ["phone", "zoom"], strict=True):
        for kind, offset, color in [
            ("mfcc-svm-control", -0.2, "#4287a8"),
            ("coatnet-reference-reproduction-attempt", 0.2, "#e18339"),
        ]:
            row = next(
                r
                for r in results
                if r["kind"] == kind
                and r["domain"] == domain
                and r["split"] == "random"
                and r["seed"] == 42
            )
            counts = np.asarray(row["cross_channel"]["confusion_matrix"]).sum(axis=0)
            axis.bar(
                np.arange(36) + offset,
                counts,
                width=0.4,
                color=color,
                label="SVM" if kind.startswith("mfcc") else "CoAtNet",
            )
        axis.set_ylabel("Jumlah prediksi")
        axis.set_title(f"Latih {domain}, uji {'zoom' if domain == 'phone' else 'phone'} (n=900)")
        axis.legend()
    axes[1].set_xticks(range(36), list(KEYS))
    axes[1].set_xlabel("Kelas yang diprediksi")
    figure.tight_layout()
    save_figure(figure, output, "transfer")


def audit_figure(manifest, raw, output):
    flagged = [f for f in manifest["files"] if f["strategy"] != "author-adaptive"]
    if not flagged or not raw:
        return
    source = flagged[0]
    wave, rate = sf.read(raw / source["path"], dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    figure, axis = plt.subplots(figsize=(13, 3))
    axis.plot(np.arange(len(wave))[::40] / rate, wave[::40], lw=0.7, color="#4287a8")
    for position in source["centers_samples"]:
        axis.axvline(position / rate, alpha=0.5, color="#e18339", lw=0.7)
    axis.set_title(
        f"Audit {source['domain']}/{source['key']}: 25 onset terpilih; "
        "posisi label asli tidak tersedia"
    )
    axis.set_xlabel("Waktu dalam rekaman asli (detik)")
    axis.set_ylabel("Amplitudo")
    figure.tight_layout()
    save_figure(figure, output, "audit-isolasi")


def keyboard_figure(output, results):
    figure, axes = plt.subplots(2, 1, figsize=(13, 8))
    rows = ["1234567890", "qwertyuiop", "asdfghjkl", "zxcvbnm"]
    positions = {
        key: (column + offset, row_number)
        for row_number, (keys, offset) in enumerate(zip(rows, [0, 0.25, 0.75, 1.25], strict=True))
        for column, key in enumerate(keys)
    }
    records = []
    for axis, domain in zip(axes, ["phone", "zoom"], strict=True):
        result = next(
            r for r in results if r["domain"] == domain and r["kind"].startswith("coatnet")
        )
        matrix = np.asarray(result["test"]["confusion_matrix"])
        for number, key in enumerate(KEYS):
            total, correct = int(matrix[number].sum()), int(matrix[number, number])
            recall = correct / total
            x, y = positions[key]
            axis.add_patch(
                Rectangle(
                    (x, y),
                    0.9,
                    0.85,
                    facecolor=plt.cm.Blues(recall),
                    edgecolor="#aaaaaa",
                    linewidth=0.7,
                )
            )
            axis.text(
                x + 0.45,
                y + 0.42,
                f"{key}\n{correct}/{total}",
                ha="center",
                va="center",
                color="white" if recall >= 0.8 else "#111111",
            )
            records.append(
                {
                    "domain": domain,
                    "key": key,
                    "support": total,
                    "correct": correct,
                    "recall": recall,
                }
            )
        pairs = [
            (int(matrix[a, b]), KEYS[a], KEYS[b])
            for a, b in np.ndindex(matrix.shape)
            if a != b and matrix[a, b] > 0
        ]
        for count, source, target in sorted(pairs, reverse=True)[:8]:
            start, end = (np.asarray(positions[key]) + [0.45, 0.4] for key in [source, target])
            axis.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="->",
                    mutation_scale=12,
                    color="#ed922b",
                    linewidth=0.8 + 0.25 * count,
                    connectionstyle="arc3,rad=.3",
                    alpha=0.8,
                )
            )
        axis.set_xlim(-0.2, 10.3)
        axis.set_ylim(4.3, -0.55)
        axis.set_aspect("equal")
        axis.axis("off")
        axis.set_title(
            f"CoAtNet {domain}: {percentage(result['test']['accuracy'])}; "
            "isi tombol = benar/5, panah = hingga 8 pasangan salah terbanyak"
        )
    figure.tight_layout()
    save_figure(figure, output, "peta-tombol")
    with (output / "per-key.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["domain", "key", "support", "correct", "recall"]
        )
        writer.writeheader()
        writer.writerows(records)


def report(root: Path, output: Path, raw: Path | None, audit_path: Path | None = None):
    record = read_json(root / "results.json")
    if record["status"] != "complete":
        raise ResearchError("Eksperimen belum lengkap; laporan akhir menunggu semua model.")
    results = record["results"]
    if sum(r["kind"].startswith("coatnet") for r in results) != 2:
        raise ResearchError("Laporan ini memerlukan CoAtNet pada kedua kanal.")
    config = read_json(root / "experiment.json")
    manifest = config["source_manifest"]
    audit = read_json(audit_path) if audit_path else None
    if audit and (
        audit["source_results_sha256"] != file_hash(root / "results.json")
        or len(audit["experiments"]) != len(results)
        or not all(r["test_and_cross_confusions_match"] for r in audit["experiments"])
    ):
        raise ResearchError("Audit checkpoint tidak cocok dengan seluruh hasil eksperimen.")
    fresh_directory(output)
    figures(root, output, results)
    audit_figure(manifest, raw, output)
    keyboard_figure(output, results)
    columns = [
        "domain",
        "model",
        "split",
        "seed",
        "n",
        "accuracy",
        "macro_f1",
        "top5",
        "cross_accuracy",
        "cross_macro_f1",
        "training_seconds",
    ]
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "domain": row["domain"],
                    "model": row["kind"],
                    "split": row["split"],
                    "seed": row["seed"],
                    "n": row["test"]["n"],
                    **{name: row["test"][name] for name in ["accuracy", "macro_f1", "top5"]},
                    "cross_accuracy": row["cross_channel"]["accuracy"],
                    "cross_macro_f1": row["cross_channel"]["macro_f1"],
                    "training_seconds": row["training_seconds"],
                }
            )
    lines = [
        "# Observasi reproduksi pada dataset eksternal",
        "",
        f"Eksperimen dimulai: {config['created_at']}. Data nyata eksternal, 36 kelas.",
        "",
        "Kajian sumber dan asumsi: [LITERATURE.md](../../research/LITERATURE.md). "
        "Ini percobaan reproduksi independen; kode pelatihan lengkap penulis "
        "tidak tersedia dalam repositori dataset.",
        "",
        "## Data dan protokol",
        "",
        f"Dataset: [Harrison 2023]({manifest['source']}), commit `{manifest['commit']}`, MIT. "
        "72 WAV menghasilkan 900 segmen per kanal, masing-masing 25 per tombol. "
        "Ponsel: 44.100 Hz stereo dirata-ratakan menjadi mono; Zoom: 32.000 Hz mono. "
        "Panjang 14.400 sampel berarti 326,5 ms dan 450 ms.",
        "",
        "Split acak terstratifikasi per kanal: 648 latihan, 72 validasi, 180 uji "
        "(18/2/5 per kelas), dilakukan sebelum augmentasi. Scaler dan normalisasi "
        "hanya menggunakan latihan. SVM memilih enam konfigurasi melalui macro-F1 "
        "validasi. CoAtNet berjalan 1.100 epoch, Adam, batch 16, lr 0,0005 turun linear; "
        "checkpoint dipilih berdasarkan akurasi validasi setiap lima epoch. "
        "Metrik uji CoAtNet dihitung setelah pelatihan selesai. Paper membedakan "
        "Random dari Stratified dan memakai Random untuk hasil final; "
        "acak terstratifikasi dalam percobaan ini adalah perbedaan protokol.",
        "",
        f"GPU: **{config['device'].get('name', config['device']['selected'])}**, "
        f"PyTorch `{config['environment']['packages']['torch']}`; profil `gfx1032`, "
        "ROCR/HIP ordinal 0. Radeon 780M tidak digunakan.",
        "",
        "Satu file, `phone/J`, memerlukan perluasan pencarian ambang ke 4,060634. "
        "Sebanyak 71 file lain memakai pencarian adaptif asli. Jumlah segmen diketahui "
        "dari protokol; ketepatan onset dan label tidak memiliki log acuan independen. "
        "Seluruh hasil tetap bersyarat pada kualitas isolasi ini.",
        "",
        "## Hasil terukur",
        "",
        "| Kanal | Model | Split / seed | Top-1 | Macro-F1 | Top-5 | Transfer kanal |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in results:
        name = "SVM" if row["kind"].startswith("mfcc") else "CoAtNet"
        lines.append(
            f"| {row['domain']} | {name} | {row['split']} / {row['seed']} | "
            f"{percentage(row['test']['accuracy'])} | {row['test']['macro_f1']:.4f} | "
            f"{percentage(row['test']['top5'])} | "
            f"{percentage(row['cross_channel']['accuracy'])} |"
        )
    lines += [
        "",
        "Top-1/Top-5: n=180; transfer: n=900 dari kanal lain, tanpa adaptasi. "
        "Peluang acak seragam adalah 1/36 = 2,78% (Top-1) dan 5/36 = 13,89% (Top-5).",
        "",
        "![Perbandingan](perbandingan.png)",
        "",
        "## Perbandingan dengan paper",
        "",
        "[Harrison dkk.](https://arxiv.org/abs/2308.01074) melaporkan 95% untuk "
        "ponsel dan 93% untuk Zoom. Selisih berikut bersifat deskriptif karena "
        "seed, split, isolasi, dan detail arsitektur asli tidak semuanya diketahui.",
        "",
    ]
    for domain, published in [("phone", 0.95), ("zoom", 0.93)]:
        baseline = [
            r["test"]["accuracy"]
            for r in results
            if r["domain"] == domain and r["kind"].startswith("mfcc") and r["split"] == "random"
        ]
        neural = next(
            r for r in results if r["domain"] == domain and r["kind"].startswith("coatnet")
        )
        low, high = neural["test"]["wilson95"]
        lines.append(
            f"- **{domain}**: SVM {percentage(np.mean(baseline))} ± "
            f"{100 * np.std(baseline, ddof=1):.2f} poin persentase (SD tiga split). "
            f"CoAtNet {neural['test']['correct']}/180 = "
            f"{percentage(neural['test']['accuracy'])}, selisih "
            f"{100 * (neural['test']['accuracy'] - published):+.2f} poin dari paper. "
            f"Wilson 95%: {percentage(low)}–{percentage(high)}. "
            f"Checkpoint epoch {neural['best_epoch']}, validasi "
            f"{percentage(neural['best_validation_accuracy'])}, "
            f"pelatihan {neural['training_seconds'] / 60:.1f} menit."
        )
    lines += [
        "",
        "SD split memakai dataset yang sama dan test set yang dapat bertumpang "
        "tindih; ini bukan variasi lintas pengguna atau sesi. CoAtNet hanya satu "
        "inisialisasi per kanal. Wilson mengasumsikan percobaan independen dan "
        "hanya memberi gambaran ketidakpastian hitungan, bukan interval generalisasi "
        "ke keyboard atau sesi baru.",
        "",
        "![Kurva pelatihan](kurva.png)",
        "",
        "## Kesalahan dan perpindahan domain",
        "",
    ]
    for domain in ["phone", "zoom"]:
        row = next(r for r in results if r["domain"] == domain and r["kind"].startswith("coatnet"))
        cm = np.asarray(row["test"]["confusion_matrix"])
        wrong = cm.copy()
        np.fill_diagonal(wrong, 0)
        pairs = [
            (int(wrong[a, b]), KEYS[a], KEYS[b])
            for a, b in np.ndindex(wrong.shape)
            if wrong[a, b] > 0
        ]
        pairs.sort(reverse=True)
        details = ", ".join(f"{a}→{b}: {n}" for n, a, b in pairs[:8]) or "Tidak ada kesalahan."
        transferred = np.asarray(row["cross_channel"]["confusion_matrix"]).sum(axis=0)
        leading = int(transferred.argmax())
        lines += [
            f"**{domain}, CoAtNet:** pasangan label→prediksi terbanyak: {details}. "
            f"Transfer kanal mencapai {percentage(row['cross_channel']['accuracy'])}; "
            f"kelas prediksi terbanyak `{KEYS[leading]}` mencakup "
            f"{int(transferred[leading])}/900 contoh.",
            "",
        ]
        if domain == "phone":
            j = KEYS.index("j")
            lines += [
                f"Pada kelas dengan isolasi diperluas (`j`), model benar {int(cm[j, j])}/5. "
                f"Jika hanya baris uji `j` dikeluarkan, akurasi deskriptif menjadi "
                f"{percentage((np.trace(cm) - cm[j, j]) / 175)} (n=175). "
                "Ini pemeriksaan sensitivitas baris uji, bukan pembenaran label "
                "atau pelatihan ulang tanpa data ambigu.",
                "",
            ]
    lines += [
        "![Peta akurasi dan kesalahan tombol](peta-tombol.png)",
        "",
        "![Confusion matrix](confusion.png)",
        "",
        "![Distribusi transfer](transfer.png)",
        "",
        "Split berurutan menahan lima ketukan terakhir untuk uji; sesi dan file "
        "sumber tetap sama. Perpindahan kanal juga mengubah sample rate, batas "
        "frekuensi mel, dan durasi potongan. Penurunan skor tidak dapat diatribusikan "
        "hanya pada codec atau jenis mikrofon.",
        "",
        "## Batasan dan tindak lanjut",
        "",
        "Eksperimen ini menguji klasifikasi tombol terisolasi dari satu pengguna dan "
        "keyboard. Kecocokan hitungan 25 bukan F1 deteksi. Dataset ini tidak mendukung "
        "pengukuran CER kalimat alami atau uji sesi independen. Akurasi tinggi pada "
        "split ini belum menjamin hasil pada dataset yang akan dikumpulkan pengguna.",
        "",
        "Tahap eksternal berikutnya yang layak: kontrol frontend dengan sample rate "
        "dan durasi seragam, audit onset manual pada subset, beberapa seed CoAtNet, "
        "lalu dataset dengan pengguna/perangkat/sesi terpisah (MKA, SKAID, atau HEAR). "
        "Setelah itu, gunakan protokol enam sesi pada keyboard pengguna; tahan sesi "
        "terakhir sampai pilihan model ditetapkan.",
        "",
        "## Artefak",
        "",
        "- `metrics.csv`: semua metrik ringkas; `results.json`: metrik lengkap dan provenance.",
        "- `perbandingan`, `kurva`, `confusion`, `transfer`, `peta-tombol`: gambar PNG/PDF.",
        "- `per-key.csv`: dukungan dan recall setiap tombol pada CoAtNet.",
        "- Direktori eksperimen menyimpan `best.pt`, normalisasi, history, split, "
        "konfigurasi, dan model SVM.",
        "",
    ]
    if raw:
        lines += ["![Audit isolasi](audit-isolasi.png)", ""]
    if audit:
        lines += [
            "## Verifikasi checkpoint",
            "",
            f"Replay tanpa pelatihan dengan PyTorch `{audit['torch']}` pada CPU "
            f"menghasilkan confusion matrix uji dan transfer yang sama persis untuk "
            f"seluruh {len(audit['experiments'])} konfigurasi. SHA-256 hasil sumber: "
            f"`{audit['source_results_sha256']}`. Prediksi per contoh, skor, dan WAV "
            "kesalahan diekspor oleh `research/replay_harrison.py`.",
            "",
        ]
    (output / "laporan.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(
        output / "results.json",
        {
            "results": results,
            "experiment": config,
            "source_results_sha256": file_hash(root / "results.json"),
            "checkpoint_audit": audit,
        },
    )
    print(output / "laporan.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=Path("data/external/harrison-2023"))
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    try:
        report(args.run, args.output, args.raw, args.audit)
    except (ResearchError, OSError, ValueError, StopIteration) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
