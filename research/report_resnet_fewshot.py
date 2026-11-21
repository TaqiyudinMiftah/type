"""Render the locked paired label-budget experiment after its CPU audit passes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ketik_audio.common import ResearchError, file_hash, fresh_directory, read_json, write_json
from ketik_audio.fewshot import KS, ROOT
from ketik_audio.resnet_analysis import analyse


def percent(stat):
    return f"{stat['mean'] * 100:.2f}% ± {stat['sd'] * 100:.2f} pp"


def run(root, audit, output, snapshot):
    record = read_json(root / "results.json")
    checked = read_json(audit / "audit.json")
    if checked["status"] != "passed" or checked["results_sha256"] != file_hash(
        root / "results.json"
    ):
        raise ResearchError("Audit hasil terkini belum lulus.")
    summary = analyse(record)
    summary.update(
        results_sha256=file_hash(root / "results.json"),
        audit_sha256=file_hash(audit / "audit.json"),
        analysis_sha256=file_hash(ROOT / "src/ketik_audio/resnet_analysis.py"),
        target_fit_labels_unique_in_campaign=checked["target_fit_labels_unique_in_campaign"],
        experiment=record,
        audit=checked,
    )
    fresh_directory(output)
    lookup = {(g["method"], g["k"]): g for g in summary["groups"]}
    colors = {"scratch": "#267a9e", "adapt": "#d57b28"}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for method in ("scratch", "adapt"):
        means = [lookup[method, k]["accuracy"]["mean"] * 100 for k in KS]
        deviations = [lookup[method, k]["accuracy"]["sd"] * 100 for k in KS]
        axes[0].errorbar(
            KS, means, yerr=deviations, marker="o", capsize=3, color=colors[method], label=method
        )
    zero = summary["zero_shot_accuracy"]
    axes[0].errorbar(
        [0],
        [zero["mean"] * 100],
        yerr=[zero["sd"] * 100],
        fmt="s",
        color="#74678c",
        capsize=3,
        label="sumber, K=0",
    )
    for threshold in (80, 90):
        axes[0].axhline(threshold, color="#aaa", linestyle="--", linewidth=0.7)
    axes[0].set_xticks((0, *KS))
    axes[0].set_xlim(-0.3, 10.5)
    axes[0].set_ylim(0, 102)
    axes[0].set_xlabel("Contoh berlabel per tombol (K)")
    axes[0].set_ylabel("Top-1 uji ponsel (%)")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_title("Rerata ± SD tiga blok")
    x = np.arange(len(KS))
    paired = summary["paired"]
    axes[1].bar(
        x,
        [p["difference"]["mean"] * 100 for p in paired],
        yerr=[p["difference"]["sd"] * 100 for p in paired],
        color="#5b9c7b",
        alpha=0.7,
        capsize=3,
        width=0.55,
    )
    for j in range(3):
        axes[1].scatter(
            x + (j - 1) * 0.11,
            [p["per_block_difference"][j] * 100 for p in paired],
            color=("#314b63", "#bd6d24", "#794975")[j],
            s=26,
            label=f"blok {j + 1}",
            zorder=3,
        )
    axes[1].axhline(0, color="#444", linewidth=0.8)
    axes[1].set_xticks(x, KS)
    axes[1].set_xlabel("Contoh berlabel per tombol (K)")
    axes[1].set_ylabel("Adaptasi − scratch (poin persentase)")
    axes[1].set_title("Selisih berpasangan; titik = tiap blok")
    axes[1].legend(loc="best", fontsize=8)
    figure.suptitle(
        "CNN residual: Zoom → ponsel, 36 kelas\n"
        "10.000 update/metode; SD mencakup pilihan dukungan dan inisialisasi",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(output / "kurva-label.png", dpi=200, bbox_inches="tight")
    figure.savefig(output / "kurva-label.pdf", bbox_inches="tight")
    plt.close(figure)
    lines = [
        "# Efisiensi label CNN residual: Zoom→ponsel",
        "",
        "Pilot satu arah memakai compact-resnet (311.892 parameter), dataset "
        "eksternal Harrison, frontend 16 kHz/400 ms yang sama dengan baseline "
        "stabil. Tiga blok menggabungkan (seed latihan, seed dukungan): "
        "(42,101), (43,102), (44,103). Dalam setiap blok/K, scratch dan adaptasi "
        "menerima dukungan, batch, augmentasi, Adam 5e-4 dan tepat 10.000 update "
        "yang sama. Perbedaan perlakuan adalah inisialisasi dari bobot sumber "
        "atau acak; optimizer target selalu baru.",
        "",
        "![Kurva efisiensi label dan selisih berpasangan](kurva-label.png)",
        "",
        "## Hasil utama",
        "",
        "| K | Label target/fit | Scratch: top-1 ± SD | Adaptasi: top-1 ± SD | "
        "Selisih adaptasi−scratch ± SD | Blok positif |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for k, paired_row in zip(KS, paired, strict=True):
        scratch, adapted = (lookup[method, k] for method in ("scratch", "adapt"))
        delta = paired_row["difference"]
        lines.append(
            f"| {k} | {36 * k} | {percent(scratch['accuracy'])} | "
            f"{percent(adapted['accuracy'])} | {delta['mean'] * 100:+.2f} ± "
            f"{delta['sd'] * 100:.2f} pp | {paired_row['positive_blocks']}/3 |"
        )
    lines += [
        "",
        f"Tanpa label target (K=0), kepala sumber: **{percent(zero)}**.",
        "",
        "SD dihitung dari tiga blok gabungan pemilihan dukungan dan "
        "inisialisasi, bukan interval keyakinan. Ketiga blok memakai "
        "360 event uji yang sama; 1.080 prediksi bukan 1.080 event independen. "
        "Efek kedua faktor variasi belum dipisahkan secara faktorial.",
        "",
        "## Ambang pada grid yang diuji",
        "",
        "| Metode | Ambang | K pertama: rerata | K pertama: semua blok |",
        "|---|---:|---:|---:|",
    ]
    for method in ("scratch", "adapt"):
        for threshold in (80, 90):
            row = summary["observed_grid_thresholds"][method][str(threshold)]
            mean_k, all_k = row["first_mean_k"], row["first_all_blocks_k"]
            lines.append(
                f"| {method} | {threshold}% | "
                f"{mean_k if mean_k is not None else 'belum sampai K=10'} | "
                f"{all_k if all_k is not None else 'belum sampai K=10'} |"
            )
    lines += [
        "",
        "Ambang ini deskriptif pada titik K=1/2/3/5/10 yang diuji, "
        "bukan estimasi jumlah label minimum yang presisi atau jaminan "
        "pada mikrofon/keyboard/sesi baru. Titik di antara grid belum "
        "diukur. Tidak memilih model/seed dari hasil uji ini.",
        "",
        "## Penguasaan dukungan dan metrik tambahan",
        "",
        "| K | Metode | Dukungan rerata ± SD | Dukungan minimum | Macro-F1 ± SD | Top-5 ± SD |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary["groups"]:
        lines.append(
            f"| {row['k']} | {row['method']} | {percent(row['support_accuracy'])} | "
            f"{row['minimum_support_accuracy'] * 100:.2f}% | "
            f"{row['macro_f1']['mean']:.4f} ± {row['macro_f1']['sd']:.4f} | "
            f"{percent(row['top5'])} |"
        )
    lines += [
        "",
        "## Model sumber dan biaya",
        "",
        "Masing-masing sumber memakai 648 train dan 72 validasi Zoom, "
        "split seed 42. Tiga pretraining dari inisialisasi 42/43/44 "
        "masing-masing 10.000 update; validasi setiap 500 update. "
        "Pilih akurasi validasi terbesar yang paling awal saat seri. "
        "Ketiga checkpoint/aturan dikunci sebelum evaluasi ponsel baru. "
        "Test sumber 180 event hanya dievaluasi setelah pilihan checkpoint.",
        "",
        "| Seed | Update terpilih | Validasi | Test sumber | Waktu pretraining |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in record["sources"]:
        lines.append(
            f"| {row['training_seed']} | {row['best_step']} | "
            f"{row['validation_accuracy'] * 100:.2f}% | "
            f"{row['test']['accuracy'] * 100:.2f}% | {row['training_seconds']:.1f} s |"
        )
    lines += [
        "",
        f"Biaya sumber unik: **720 label** (648 train +72 validasi), "
        "dipakai ulang pada ketiga seed. Anggaran target positif per fit "
        "adalah **36K label** dan **nol validasi target**; biaya sumber "
        "tetap harus dihitung ketika menilai manfaat adaptasi. "
        "Arsitektur bersama sebelumnya dipilih memakai 432 label Zoom "
        "yang tercakup dalam 720 label ini; scratch juga memakai pilihan "
        "arsitektur tersebut, sehingga biaya pemilihan model historis "
        "tidak boleh dianggap nol. "
        f"Seluruh pilihan dukungan dalam kampanye ini mencakup "
        f"{summary['target_fit_labels_unique_in_campaign']} label target "
        "unik, dipakai ulang pada beberapa fit/grid. Itu berbeda dari "
        "anggaran satu model yang hanya menerima dukungannya sendiri.",
        "",
        f"Total waktu pretraining tiga sumber: "
        f"{summary['source_training_seconds_total'] / 60:.2f} menit; "
        f"total 30 fit target: {summary['target_training_seconds_total'] / 60:.2f} menit. "
        "Semua memakai RX 6600 LE. Anggaran update target sama; sumber "
        "memberi biaya komputasi dan supervisi tambahan sebelum adaptasi.",
        "",
        "## Semua hasil target",
        "",
        "| Blok | Seed | Seed dukungan | K | Metode | Dukungan | Top-1 uji | Benar/uji |",
        "|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    per_run = []
    for row in record["conditions"]:
        support = f"{row['support_accuracy'] * 100:.2f}%" if row["k"] else "—"
        draw = row["support_seed"] if row["support_seed"] is not None else "—"
        lines.append(
            f"| {row['block']} | {row['training_seed']} | {draw} | {row['k']} | "
            f"{row['method']} | {support} | {row['test']['accuracy'] * 100:.2f}% | "
            f"{row['test']['correct']}/{row['test']['n']} |"
        )
        per_run.append(
            {
                "block": row["block"],
                "seed": row["training_seed"],
                "support_seed": row["support_seed"],
                "k": row["k"],
                "method": row["method"],
                "accuracy": row["test"]["accuracy"],
                "macro_f1": row["test"]["macro_f1"],
                "top5": row["test"]["top5"],
                "support_accuracy": row.get("support_accuracy"),
                "correct": row["test"]["correct"],
                "n": row["test"]["n"],
                "training_seconds": row.get("training_seconds"),
            }
        )
    lines += [
        "",
        "## Audit dan batas interpretasi",
        "",
        "33 checkpoint fit baru (tiga sumber dan 30 target), ditambah "
        "tiga evaluasi K=0, direplay di CPU. Audit memeriksa prediksi "
        "individual, metrik, ID/anggaran, statistik hanya dari fit, "
        "inisialisasi sumber/acak, counter Adam/BatchNorm, jadwal LR dan "
        "fingerprint batch/augmentasi berpasangan. Tidak membawa state "
        "optimizer sumber ke target. Checkpoint pemulihan tersedia untuk "
        "model, optimizer, scheduler dan RNG.",
        "",
        "Selisih skor test absolut maksimum CPU/GPU pada 36 evaluasi: "
        f"{max(e['max_absolute_score_difference'] for e in checked['evaluations']):.6g}.",
        "",
        "Kondisi K>0 menormalisasi input dari dukungan target; K=0 memakai "
        "statistik sumber. Jadi perubahan K=0→K>0 mencakup normalisasi dan "
        "fine-tuning. Selisih adaptasi−scratch memakai normalisasi target "
        "yang sama dan mengukur manfaat inisialisasi pada satu resep tetap. "
        "Hasil ini tidak memisahkan pengaruh komponen model atau menetapkan "
        "efektivitas seluruh strategi adaptasi.",
        "",
        "Pemisahan masih dalam WAV Harrison yang sama. Uji ponsel telah "
        "dilihat pada tahap sebelumnya, arah diagnosis pernah memakai "
        "hasil ponsel historis, dan arsitektur telah dipilih melalui "
        "validasi Zoom. Ini pilot eksploratif, bukan holdout baru, SOTA "
        "atau pengujian sesi/keyboard/partisipan baru. Isolasi berbantuan "
        "jumlah ketukan dan ketidakpastian phone/J masih berlaku. Hasil "
        "adalah klasifikasi potongan ketukan, belum deteksi/teks bebas.",
        "",
        "SD baseline repair K=10 sebelumnya berasal dari tiga "
        "inisialisasi dengan dukungan tetap seed 101. SD di sini mencakup "
        "dukungan sekaligus inisialisasi sehingga tidak dapat ditafsirkan "
        "sebagai perubahan kestabilan inisialisasi saja.",
        "",
        "Protokol: `research/RESNET_FEWSHOT_PROTOCOL.md`. "
        f"SHA256 hasil: `{summary['results_sha256']}`.",
        "",
    ]
    (output / "laporan.md").write_text("\n".join(lines), encoding="utf-8")
    with (output / "per-run.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_run[0]))
        writer.writeheader()
        writer.writerows(per_run)
    write_json(output / "summary.json", summary)
    if snapshot:
        if snapshot.exists():
            raise ResearchError("Snapshot sudah ada.")
        write_json(snapshot, summary)
    print(f"Laporan tersimpan: {output / 'laporan.md'}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/resnet-fewshot")
    parser.add_argument("--audit", type=Path, default=ROOT / "reports/resnet-fewshot-audit")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/resnet-fewshot")
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    try:
        run(args.run, args.audit, args.output, args.snapshot)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
