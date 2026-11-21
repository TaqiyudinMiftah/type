"""Report frozen BEATs/ResNet probes, every heldout folder and support budget."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, read_json, write_json
from ketik_audio.mka import PLATFORMS
from ketik_audio.mka_lopo import CACHE, ROOT
from ketik_audio.mka_lopo_analysis import analyse


def percent(value):
    return f"{100 * value:.2f}".replace(".", ",")


def display(record):
    value = f"{percent(record['mean'])}%"
    return value if record["sd"] is None else f"{value} ± {percent(record['sd'])} pp"


def figures(analysis, output):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    x = np.arange(6)
    rows = analysis["per_folder_full_budget"]
    axes[0].scatter(
        x - 0.1,
        [100 * r["beats"]["test"]["accuracy"]["mean"] for r in rows],
        color="#2563a6",
        label="BEATs: satu encoder",
        s=38,
        zorder=3,
    )
    axes[0].errorbar(
        x + 0.1,
        [100 * r["resnet"]["test"]["accuracy"]["mean"] for r in rows],
        yerr=[100 * r["resnet"]["test"]["accuracy"]["sd"] for r in rows],
        fmt="o",
        color="#c3611c",
        capsize=4,
        label="ResNet: mean ± SD 3 seed",
    )
    for index, row in enumerate(rows):
        axes[0].scatter(
            index + 0.1 + np.linspace(-0.03, 0.03, 3),
            np.array(row["resnet"]["test"]["accuracy"]["values"]) * 100,
            color="#c3611c",
            s=13,
            alpha=0.5,
        )
    axes[0].set_xticks(x, PLATFORMS, rotation=20)
    axes[0].set_title("K=6: 1.080 label dari lima folder latih", fontsize=11)
    axes[0].set_xlabel("Folder uji: 216 potongan dari 36 WAV")
    axes[0].set_ylim(0, 8)
    curve = analysis["curve"]
    budgets = [r["fit_labels_per_fold"] for r in curve]
    for family, color, label in (
        ("beats", "#2563a6", "BEATs: SD 3 draw (K6 satu set)"),
        ("resnet", "#c3611c", "ResNet: SD 3 mean inisialisasi"),
    ):
        means = [100 * r[family]["test"]["accuracy"]["mean"] for r in curve]
        deviation = [
            100
            * (
                (r["beats"]["test"]["accuracy"]["sd"] or 0)
                if family == "beats"
                else r["resnet_source_mean_accuracy"]["sd"]
            )
            for r in curve
        ]
        axes[1].errorbar(
            budgets, means, yerr=deviation, fmt="o-", color=color, capsize=4, label=label
        )
    axes[1].set_xticks(
        budgets,
        [
            f"{b}\nK={r['k_per_key_per_training_folder']}"
            for b, r in zip(budgets, curve, strict=True)
        ],
    )
    axes[1].set_xlabel("Label latih per fold; K per tombol per folder latih")
    axes[1].set_title("Prediksi out-of-fold: 1.296 potongan unik", fontsize=11)
    axes[1].set_ylim(0, 8)
    for axis in axes:
        axis.axhline(100 / 36, linestyle="--", color="0.4", linewidth=1, label="Peluang 1/36")
        axis.set_ylabel("Akurasi top-1 (%)")
        axis.legend(fontsize=7.5, loc="upper left")
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("BEATs dan ResNet beku: transfer lintas folder MKA", fontsize=14)
    fig.text(
        0.5,
        0.015,
        "Error bar deskriptif, bukan CI sesi. Setiap fold menahan seluruh WAV satu folder.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(output / "lintas-folder.png", dpi=180)
    fig.savefig(output / "lintas-folder.pdf")
    plt.close(fig)


def run(root, cache, audit_dir, output, snapshot):
    result = read_json(root / "results.json")
    cache_record = read_json(cache / "manifest.json")
    audit = read_json(audit_dir / "audit.json")
    if (
        audit["status"] != "passed"
        or audit["run_results_sha256"] != file_hash(root / "results.json")
        or audit["cache_manifest_sha256"] != file_hash(cache / "manifest.json")
    ):
        raise ResearchError("Audit tidak sesuai run/cache.")
    for directory, record in ((root, result), (cache, cache_record), (audit_dir, audit)):
        for name, digest in record["artifact_sha256"].items():
            if file_hash(directory / name) != digest:
                raise ResearchError(f"Artefak berubah: {directory / name}")
    analysis = analyse(root, result)
    if snapshot and snapshot.exists():
        raise ResearchError("Snapshot sudah ada.")
    fresh_directory(output)
    figures(analysis, output)
    csv_rows = []
    for row in result["conditions"]:
        entry = {
            key: row[key]
            for key in (
                "id",
                "heldout",
                "encoder",
                "family",
                "source_seed",
                "k",
                "draw",
                "permuted",
                "support_count",
                "support_parents",
                "support_accuracy",
            )
        }
        for unit in ("test", "parent"):
            entry.update(
                {
                    f"{unit}_{key}": row[unit][key]
                    for key in (
                        "n",
                        "correct",
                        "accuracy",
                        "top5",
                        "macro_f1",
                        "predicted_class_count",
                    )
                }
            )
        csv_rows.append(entry)
    with (output / "per-condition.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    train_summary = []
    for k in (1, 2, 3, 5, 6):
        entry = {"k": k}
        for family in ("beats", "resnet"):
            values = [
                r["support_accuracy"]
                for r in result["conditions"]
                if r["k"] == k and r["family"] == family and not r["permuted"]
            ]
            entry[family] = {
                "mean": float(np.mean(values)),
                "minimum": min(values),
                "maximum": max(values),
            }
        train_summary.append(entry)
    full = next(r for r in analysis["curve"] if r["k_per_key_per_training_folder"] == 6)
    negative_beats = display(analysis["negative_control"]["beats"]["test"]["accuracy"])
    negative_resnet = display(analysis["negative_control"]["resnet"]["test"]["accuracy"])
    lines = [
        "# BEATs vs ResNet lintas folder MKA: 8 November 2026",
        "",
        "**BEATs beku belum memperbaiki transfer lintas folder pada protokol ini.** "
        f"Dengan semua contoh latih, BEATs mencapai {display(full['beats']['test']['accuracy'])}; "
        f"ResNet {display(full['resnet']['test']['accuracy'])}. "
        "Peluang uniform adalah 2,78%. Kedekatan dengan peluang bersifat deskriptif; "
        "ini bukan uji kesetaraan statistik atau bukti bahwa sinyal tombol tidak ada.",
        "",
        "Enam fold memakai lima folder MKA untuk melatih kepala Ridge alpha1 dan "
        "menahan seluruh folder keenam untuk uji. Semua encoder tetap beku, input "
        "identik (crop peak400ms studi sebelumnya), serta label/indeks identik "
        "antar representasi. K=6 memakai 1.080 label potongan dari 180 WAV induk; "
        "folder uji memiliki 216 potongan dari 36 WAV yang terpisah. "
        "Ini belajar dari folder MKA lain, berbeda dari zero-shot Harrison → MKA.",
        "",
        "## Seluruh folder uji, K=6",
        "",
        "BEATs memiliki satu encoder tetap. ResNet melaporkan tiga inisialisasi "
        "sumber Harrison, tanpa memilih yang terbaik atau membuat ensemble. "
        "SD antar inisialisasi bukan variasi sesi/keyboard.",
        "",
        "| Folder uji | BEATs top-1 | ResNet top-1, mean ± SD | BEATs − mean ResNet |",
        "|---|---:|---:|---:|",
    ]
    for row in analysis["per_folder_full_budget"]:
        lines.append(
            f"| {row['heldout']} | {display(row['beats']['test']['accuracy'])} | "
            f"{display(row['resnet']['test']['accuracy'])} | "
            f"{100 * row['beats_minus_resnet_mean_accuracy']:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "Gabungan out-of-fold memuat 1.296 potongan unik, setiap potongan "
            "menjadi uji tepat sekali per konfigurasi. ResNet per seed mencapai "
            + ", ".join(percent(v) + "%" for v in full["resnet"]["test"]["accuracy"]["values"])
            + ". Fold berbagi data latih, sehingga tidak dianggap enam sesi independen.",
            "",
            "![Hasil lintas folder dan kurva label](lintas-folder.png)",
            "",
            "## Jumlah contoh dan variasi",
            "",
            "K adalah jumlah contoh per tombol PADA SETIAP lima folder latih. K=1 "
            "berarti lima contoh/tombol atau 180 label total per fold, bukan satu "
            "label/tombol total. Tidak ada label dari folder uji untuk fit/seleksi. "
            "Semua K tetap memakai 180 WAV induk latih.",
            "",
            "| K/folder | Label/fit | BEATs top-1 | ResNet top-1 | Selisih berpasangan |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["curve"]:
        lines.append(
            f"| {row['k_per_key_per_training_folder']} | {row['fit_labels_per_fold']} | "
            f"{display(row['beats']['test']['accuracy'])} | "
            f"{display(row['resnet']['test']['accuracy'])} | "
            f"{100 * row['paired_draw_difference']['mean']:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "SD BEATs K<6 berasal dari tiga draw dukungan. SD ResNet pada tabel "
            "K<6 adalah sebaran deskriptif sembilan kombinasi (tiga inisialisasi × "
            "tiga draw); faktor-faktor ini dipisahkan dalam JSON. Gambar kanan "
            "menampilkan SD tiga mean inisialisasi ResNet setelah merata-ratakan "
            "draw, bukan SD sembilan kombinasi tersebut. K6 hanya mempunyai satu "
            "set dukungan unik: BEATs tidak diberi SD palsu; SD ResNet K6 dari tiga "
            "inisialisasi. Selisih per draw memakai satu BEATs dikurangi mean tiga "
            "ResNet, bukan sembilan ulangan BEATs independen. Tidak ada pola kenaikan "
            "akurasi uji yang konsisten saat jumlah contoh bertambah.",
            "",
            "## Kemampuan mempelajari data latih",
            "",
            "| K/folder | BEATs rerata akurasi dukungan | ResNet rerata akurasi dukungan |",
            "|---:|---:|---:|",
        ]
    )
    for row in train_summary:
        lines.append(
            f"| {row['k']} | {percent(row['beats']['mean'])}% | {percent(row['resnet']['mean'])}% |"
        )
    lines.extend(
        [
            "",
            "BEATs menyesuaikan data latih jauh lebih baik: 92,41% pada K1 dan "
            "70,32% pada K6, tetapi prediksi folder baru tetap rendah. Informasi "
            "untuk membedakan contoh latih belum menjadi representasi tombol yang "
            "berpindah antar folder. Sinyal yang dipakai dapat terkait rekaman. "
            "Akurasi latih belum 100%, terutama ResNet; alpha1 tetap dan backbone "
            "beku belum menguji seluruh kapasitas model atau pilihan optimasi.",
            "",
            "## Kontrol negatif dan agregasi WAV",
            "",
            "Kontrol mengacak label pada tingkat WAV induk latih, mempertahankan "
            "enam potongan satu induk dengan label yang sama dan 30 potongan/kelas. "
            f"Skor gabungan: BEATs {negative_beats}; ResNet {negative_resnet}. "
            "Kontrol tidak menunjukkan pemisahan yang berguna dari model berlabel "
            "asli pada tingkat akurasi uji ini. Ini satu permutasi tetap, bukan "
            "distribusi null atau uji signifikansi.",
            "",
            "Agregasi enam skor potongan ke satu prediksi WAV juga rendah: "
            f"BEATs {display(full['beats']['parent']['accuracy'])}; "
            f"ResNet {display(full['resnet']['parent']['accuracy'])}, "
            "pada 216 WAV out-of-fold. Agregasi memakai label kelompok yang sama, "
            "bukan decoding teks atau event tambahan independen.",
            "",
            "## Audit dan biaya",
            "",
            "Seluruh 312 fit utama/sensitivitas dan 24 kontrol negatif selesai. "
            "Auditor terpisah merekonstruksi crop dari ZIP, menolak induk/PCM "
            "yang melintasi fit/uji, mengulang seluruh embedding di CPU dan "
            "seluruh kepala, serta menyelesaikan persamaan ridge terpusat secara "
            "langsung. Semua koefisien/skor memenuhi toleransi yang ditetapkan, "
            "prediksi/metrik cocok dan semua state encoder tetap sama.",
            "",
            f"Audit mengekspor {audit['classification_rows']:,} prediksi potongan/WAV. "
            f"Perbedaan fitur maksimum BEATs GPU/CPU "
            f"{audit['feature_comparisons'][0]['maximum_absolute_difference']:.8f}; "
            "ResNet identik. Tidak ada perubahan prediksi dari embedding CPU "
            f"pada {audit['heads_verified']} kondisi.",
            "",
            "BEATs: 90.311.792 parameter, 768 fitur. ResNet: 311.892 parameter "
            "termasuk head sumber, 128 fitur. ResNet menggunakan 720 label Harrison "
            "unik untuk pretraining/validasi; BEATs memakai pretraining audio "
            "berbeda dan biaya tersebut tidak disetarakan. Tidak menafsirkan "
            "perbandingan sebagai efek kausal ukuran/arsitektur saja.",
            "",
            f"BEATs diekstrak pada {cache_record['encoders'][0]['hardware']['name']} "
            f"dalam {cache_record['encoders'][0]['extraction_and_control_seconds']:.2f} detik "
            "termasuk kontrol sumber, di luar waktu load/hash/probe. ResNet diekstrak "
            f"pada CPU; fit semua kepala {result['seconds']:.2f} detik; audit CPU "
            f"{audit['seconds']:.2f} detik. Radeon 780M tidak digunakan. "
            "Tidak melatih ulang backbone atau mengunduh dataset tambahan.",
            "",
            "## Kesimpulan dan batas",
            "",
            "Mengganti representasi beku menjadi BEATs belum menyelesaikan "
            "generalisasi lintas folder MKA. Hasil tidak mendukung klaim bahwa "
            "model lebih besar otomatis lebih baik. Ini hasil dua pipeline "
            "terpilih dengan Ridge alpha1 dan crop peak400ms, bukan peringkat SOTA.",
            "",
            "Seluruh folder sudah diperiksa sebelumnya; cross-validation ini "
            "eksploratif dan semua 1.296 label dipakai di fold lain. Nama folder "
            "bukan ID fisik/sesi. Pemisahan file/PCM belum membuktikan independensi "
            "provenance dari semua transformasi. Perubahan kanal, keyboard, "
            "cara mengetik dan segmentasi belum dapat dipisahkan secara kausal. "
            "Satu WAV/tombol/folder tetap belum mendukung kalibrasi dan uji "
            "lintas rekaman pada folder yang sama.",
            "",
            "Langkah berikut yang layak adalah diagnostik pemilihan kepala "
            "dan kalibrasi: periksa apakah pengaturan linear yang dipilih hanya "
            "dari folder latih dapat meningkatkan transfer, atau apakah label "
            "dari domain target memang diperlukan. Pertanyaan efisiensi "
            "kalibrasi ke sesi baru tetap membutuhkan rekaman target terpisah. "
            "Tidak memilih alpha/checkpoint baru dari hasil uji studi ini.",
            "",
            "## Artefak",
            "",
            "- [Protokol sebelum ekstraksi/fit](../../research/MKA_LOPO_PROTOCOL.md).",
            "- [336 kondisi](per-condition.csv), [ringkasan JSON](summary.json).",
            "- [Lock dan split](../../runs/mka-lopo/lock.json), "
            "[seluruh koefisien/skor](../../runs/mka-lopo/conditions/).",
            "- [Audit CPU](../mka-lopo-audit/audit.json), "
            "[prediksi individual](../mka-lopo-audit/classifications.csv).",
            "- [Gambar PDF](lintas-folder.pdf), "
            "[observasi riset](../../research/MKA_LOPO_OBSERVATIONS.md).",
            "",
        ]
    )
    (output / "laporan.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "created_at": now(),
        "status": "complete",
        "analysis": analysis,
        "support_accuracy": train_summary,
        "run_results_sha256": file_hash(root / "results.json"),
        "cache_manifest_sha256": file_hash(cache / "manifest.json"),
        "audit_sha256": file_hash(audit_dir / "audit.json"),
        "report_code_sha256": file_hash(Path(__file__)),
        "analysis_code_sha256": file_hash(ROOT / "src/ketik_audio/mka_lopo_analysis.py"),
        "artifact_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir())},
    }
    write_json(output / "summary.json", summary)
    if snapshot:
        write_json(snapshot, summary)
    print(f"Reported all 336 conditions: {output / 'laporan.md'}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/mka-lopo")
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--audit", type=Path, default=ROOT / "reports/mka-lopo-audit")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/mka-lopo")
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    try:
        run(args.run, args.cache, args.audit, args.output, args.snapshot)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
