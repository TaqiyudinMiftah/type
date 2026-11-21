"""Report all platforms, seeds and prespecified MKA zero-shot crop policies."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, read_json, write_json
from ketik_audio.mka import KEYS, PLATFORMS
from ketik_audio.mka_zeroshot import POLICIES, ROOT, SEEDS


def percent(value):
    return f"{100 * value:.2f}".replace(".", ",")


def mean_sd(record):
    return f"{percent(record['mean'])}% ± {percent(record['sd'])} pp"


def plot(groups, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(PLATFORMS))
    labels = ("Peak: utama", "Tiled: sensitivitas")
    colors = ("#2563a6", "#c3611c")
    for axis, unit, title, limit in zip(
        axes,
        ("clip", "parent"),
        ("Per potongan: 216/platform", "Rata-rata 6 potongan: 36 WAV/platform"),
        (8, 12),
        strict=True,
    ):
        for policy, offset, label, color in zip(
            POLICIES, (-0.12, 0.12), labels, colors, strict=True
        ):
            selected = [
                next(g for g in groups if g["policy"] == policy and g["platform"] == p)
                for p in PLATFORMS
            ]
            means = [100 * g[unit]["accuracy"]["mean"] for g in selected]
            sd = [100 * g[unit]["accuracy"]["sd"] for g in selected]
            axis.errorbar(x + offset, means, yerr=sd, fmt="o", color=color, capsize=4, label=label)
            for index, group in enumerate(selected):
                axis.scatter(
                    index + offset + np.linspace(-0.035, 0.035, 3),
                    np.array(group[unit]["accuracy"]["per_seed"]) * 100,
                    s=12,
                    color=color,
                    alpha=0.55,
                )
        axis.axhline(100 / 36, color="0.35", linestyle="--", linewidth=1, label="Peluang 1/36")
        axis.set_xticks(x, PLATFORMS, rotation=20)
        axis.set_ylim(0, limit)
        axis.set_ylabel("Akurasi top-1 (%)")
        axis.set_title(title, fontsize=11)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Harrison Zoom → MKA: tanpa kalibrasi target", fontsize=14)
    fig.text(
        0.5,
        0.02,
        "Titik kecil: 3 seed sumber. Galat: sample SD, bukan CI sesi. Skala y kedua panel berbeda.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(output / "transfer.png", dpi=180)
    fig.savefig(output / "transfer.pdf")
    plt.close(fig)


def run(root, audit_dir, output, snapshot):
    result = read_json(root / "results.json")
    audit = read_json(audit_dir / "audit.json")
    if audit["status"] != "passed" or audit["run_results_sha256"] != file_hash(
        root / "results.json"
    ):
        raise ResearchError("Audit tidak cocok dengan run.")
    for folder, record in ((root, result), (audit_dir, audit)):
        for name, digest in record["artifact_sha256"].items():
            if file_hash(folder / name) != digest:
                raise ResearchError(f"Artefak berubah: {folder / name}")
    if snapshot and snapshot.exists():
        raise ResearchError("Snapshot sudah ada.")
    fresh_directory(output)
    records = read_json(root / "records.json")["records"]
    groups = result["groups"]
    plot(groups, output)
    csv_rows = []
    for condition in result["conditions"]:
        row = {key: condition[key] for key in ("platform", "policy", "seed")}
        for unit in ("clip", "parent"):
            row.update(
                {
                    f"{unit}_{key}": condition[unit][key]
                    for key in (
                        "n",
                        "correct",
                        "accuracy",
                        "top5",
                        "macro_f1",
                        "predicted_class_count",
                        "most_frequent_prediction",
                        "most_frequent_prediction_count",
                    )
                }
            )
        csv_rows.append(row)
    with (output / "per-condition.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    source_controls = result["source_controls"]
    clamps = {p: sum(r["peak_clamped"] for r in records if r["platform"] == p) for p in PLATFORMS}
    summary = {
        "created_at": now(),
        "status": "complete",
        "source": "Harrison Zoom",
        "target": "MKA v3",
        "unit": "supplied excerpt, not an annotated keystroke",
        "seeds": list(SEEDS),
        "class_order": KEYS,
        "unique_clips": result["clips"],
        "unique_parent_wavs": result["parent_groups"],
        "target_fit_labels": 0,
        "training_updates": 0,
        "gpu_used": False,
        "groups": groups,
        "source_controls": source_controls,
        "peak_clamped_per_platform": clamps,
        "variation_scope": "sample SD over three source initializations, not sessions",
        "run_results_sha256": file_hash(root / "results.json"),
        "audit_sha256": file_hash(audit_dir / "audit.json"),
        "code_sha256": file_hash(Path(__file__)),
    }
    lines = [
        "# Transfer langsung Harrison → MKA: hasil 8 November 2026",
        "",
        "Classifier compact-resnet sumber **belum dapat digunakan langsung pada MKA**: "
        "rerata akurasi utama per platform hanya 2,62–4,63%, dekat baseline peluang "
        "uniform 2,78%. Ini deskripsi hasil, bukan uji kesetaraan statistik dengan peluang.",
        "",
        "Seluruh 1.296 potongan unik, 216 WAV induk, enam platform, 36 kelas dan tiga "
        "seed sumber dilaporkan. Tidak ada fit target, update parameter, statistik target "
        "atau GPU yang digunakan. Sumber dilatih sebelumnya dengan 648 train + 72 "
        "validasi label unik. Model ini baseline riset, belum terbukti SOTA.",
        "",
        "## Hasil per potongan",
        "",
        "Rerata ± sample SD antar tiga inisialisasi sumber; satu platform memiliki "
        "216 potongan dari 36 WAV. Potongan dalam WAV saling bergantung. SD ini tidak "
        "mengukur variasi sesi, keyboard atau pengguna.",
        "",
        "| Platform | Peak utama, top-1 | Tiled sensitivitas, top-1 | Peak top-5 | Peak macro-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for platform in PLATFORMS:
        primary = next(g for g in groups if g["platform"] == platform and g["policy"] == "peak")
        tiled = next(g for g in groups if g["platform"] == platform and g["policy"] == "tiled")
        lines.append(
            f"| {platform} | {mean_sd(primary['clip']['accuracy'])} | "
            f"{mean_sd(tiled['clip']['accuracy'])} | {mean_sd(primary['clip']['top5'])} | "
            f"{primary['clip']['macro_f1']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Peluang uniform top-5 = 13,89%. Tiled juga rendah (1,54–4,17% top-1); "
            "kebijakan utama tidak diganti setelah melihat hasil. Setiap potongan "
            "menyumbang satu prediksi, meskipun tiled memakai empat jendela. "
            "Tidak menggabungkan tiga seed menjadi ensemble.",
            "",
            "![Transfer langsung](transfer.png)",
            "",
            "## Agregasi WAV induk",
            "",
            "Rata-rata enam logit potongan per WAV, kemudian klasifikasi satu label. "
            "Ada 36 WAV/platform (satu per tombol), bukan tambahan contoh independen.",
            "",
            "| Platform | Peak, top-1 WAV | Tiled, top-1 WAV |",
            "|---|---:|---:|",
        ]
    )
    for platform in PLATFORMS:
        selected = [
            next(g for g in groups if g["platform"] == platform and g["policy"] == policy)
            for policy in POLICIES
        ]
        lines.append(
            f"| {platform} | {mean_sd(selected[0]['parent']['accuracy'])} | "
            f"{mean_sd(selected[1]['parent']['accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "## Pemeriksaan kegagalan transfer",
            "",
            "Pada kebijakan peak, model hanya pernah memprediksi 2–14 dari 36 kelas "
            "untuk satu platform/seed. Pada hp seed44, 213/216 potongan diprediksi '8'; "
            "sensitivitas tiled pada pasangan yang sama memprediksi '8' untuk 216/216. "
            "Bias prediksi ini sesuai dengan lemahnya transfer classifier sumber, "
            "tetapi belum membuktikan penyebab tunggal berupa gain, codec, atau keyboard.",
            "",
            "| Platform | Crop peak di-clamp ke batas potongan /216 |",
            "|---|---:|",
        ]
    )
    for platform in PLATFORMS:
        lines.append(f"| {platform} | {clamps[platform]} |")
    lines.extend(
        [
            "",
            f"Total {result['peak_clamped']}/1.296 crop memerlukan clamp. "
            "Peak adalah puncak amplitudo, bukan anotasi onset. Potongan MKA dapat "
            "berisi beberapa transien; definisinya berbeda dari ketukan Harrison "
            "yang diisolasi dengan bantuan jumlah event. Dua kebijakan yang telah "
            "ditentukan tetap sama-sama buruk; ini belum menyingkirkan semua pengaruh segmentasi.",
            "",
            "## Kontrol dan audit",
            "",
            "| Seed sumber | Holdout Harrison Zoom, n=180 | Harrison phone K=0, n=360 |",
            "|---|---:|---:|",
        ]
    )
    for seed in SEEDS:
        selected = [
            next(r for r in source_controls if r["seed"] == seed and r["domain"] == domain)
            for domain in ("zoom", "phone")
        ]
        lines.append(
            f"| {seed} | {percent(selected[0]['accuracy'])}% | "
            f"{percent(selected[1]['accuracy'])}% |"
        )
    lines.extend(
        [
            "",
            "Prediksi individual kedua kontrol identik dengan eksperimen lama, "
            "baik menggunakan cache fitur lama maupun frontend CPU yang dihitung ulang. "
            "Selisih maksimum fitur CPU/cache GPU 0,001786 dB dan logit CPU/GPU lama "
            "0,00002289. Tidak ada perbedaan prediksi. Ini memeriksa konsistensi "
            "implementasi; domain MKA tetap mempunyai perubahan data dan segmentasi sekaligus.",
            "",
            "Audit merekonstruksi 1.296 crop peak dan 5.184 jendela tiled langsung "
            "dari ZIP, memeriksa seluruh fitur, mengulang enam kombinasi seed/kebijakan "
            "dan enam kontrol sumber, serta mengekspor 9.072 prediksi potongan/WAV. "
            "Seluruh logit replay CPU identik, metrik cocok, bobot dan buffer BatchNorm "
            "tetap sama. Metrik tidak memakai interval binomial per potongan.",
            "",
            f"Pemrosesan awal {result['prepare_seconds']:.1f} detik; inferensi dan kontrol "
            f"{sum(r['inference_and_controls_seconds'] for r in result['timings']):.1f} detik "
            f"pada CPU empat thread. Audit {audit['seconds']:.1f} detik. "
            "Tidak memakai Radeon 780M atau menjalankan fit GPU baru.",
            "",
            "## Arti dan batas",
            "",
            "Akurasi tinggi dalam Harrison tidak membuktikan kesiapan lintas dataset. "
            "Hasil ini menetapkan baseline zero-shot yang lemah; belum menyimpulkan "
            "bahwa adaptasi berlabel akan gagal atau bahwa model yang lebih besar "
            "akan menyelesaikan masalah. Tidak menguji decoding kalimat atau "
            "penentuan jumlah ketukan dari rekaman bebas.",
            "",
            "Nama platform bukan ID fisik/sesi yang terverifikasi. Audit menolak "
            "salinan persis Harrison tetapi belum mencari semua salinan yang "
            "dipotong/resample/ditransformasi. MKA tetap belum mendukung "
            "kalibrasi dan uji lintas rekaman pada platform yang sama karena "
            "hanya satu WAV per tombol/platform.",
            "",
            "Langkah berikut yang layak adalah membandingkan representasi sumber "
            "dengan embedding audio pralatih pada protokol yang memisahkan WAV "
            "induk. Evaluasi lintas folder platform MKA dapat menjadi pertanyaan "
            "tambahan; uji efisiensi kalibrasi ke sesi baru tetap memerlukan "
            "rekaman target terpisah. Pilihan representasi/split tersebut belum diuji di sini.",
            "",
            "## Artefak",
            "",
            "- [Protokol sebelum inferensi](../../research/MKA_ZEROSHOT_PROTOCOL.md).",
            "- [Lock, crop, fitur dan skor lengkap](../../runs/mka-zeroshot/).",
            "- [Audit replay](../mka-zeroshot-audit/audit.json) dan "
            "[prediksi individual](../mka-zeroshot-audit/classifications.csv).",
            "- [Seluruh 36 kondisi](per-condition.csv), [ringkasan JSON](summary.json), "
            "[gambar PDF](transfer.pdf).",
            "- [Audit data dan batas provenance](../../research/MKA_OBSERVATIONS.md).",
            "",
        ]
    )
    (output / "laporan.md").write_text("\n".join(lines), encoding="utf-8")
    summary["artifact_sha256"] = {p.name: file_hash(p) for p in sorted(output.iterdir())}
    write_json(output / "summary.json", summary)
    if snapshot:
        write_json(snapshot, summary)
    print(f"Reported all {len(result['conditions'])} conditions: {output / 'laporan.md'}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/mka-zeroshot")
    parser.add_argument("--audit", type=Path, default=ROOT / "reports/mka-zeroshot-audit")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/mka-zeroshot")
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    try:
        run(args.run, args.audit, args.output, args.snapshot)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
