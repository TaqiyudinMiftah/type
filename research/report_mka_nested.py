"""Report source-only nested alpha selection and its paired fixed-alpha baseline."""

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
from ketik_audio.mka_lopo import ROOT
from ketik_audio.mka_nested import ALPHAS
from ketik_audio.mka_nested_analysis import analyse


def percent(value):
    return f"{100 * value:.2f}".replace(".", ",")


def display(value):
    text = f"{percent(value['mean'])}%"
    return text if value["sd"] is None else f"{text} ± {percent(value['sd'])} pp"


def plots(analysis, selection, output):
    rows = [r for r in analysis["per_folder"] if not r["permuted"]]
    x = np.arange(6)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, family, title in zip(
        axes,
        ("beats", "resnet"),
        ("BEATs: satu encoder", "ResNet: tiga inisialisasi sumber"),
        strict=True,
    ):
        for key, offset, color, label in (
            ("baseline_accuracy", -0.1, "#777777", "Baseline α=1"),
            ("nested_accuracy", 0.1, "#2563a6", "α dari validasi internal"),
        ):
            means = [100 * row[family][key]["mean"] for row in rows]
            deviations = [100 * (row[family][key]["sd"] or 0) for row in rows]
            axis.errorbar(
                x + offset, means, yerr=deviations, fmt="o", capsize=4, color=color, label=label
            )
        axis.axhline(100 / 36, linestyle="--", color="0.4", linewidth=1, label="Peluang 1/36")
        axis.set_xticks(x, PLATFORMS, rotation=20)
        axis.set_ylim(0, 8)
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Folder uji luar: 216 potongan/folder")
        axis.set_ylabel("Akurasi top-1 (%)")
        axis.legend(fontsize=8, loc="upper left")
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Regularisasi bersarang belum memperbaiki transfer MKA", fontsize=14)
    fig.text(
        0.5,
        0.02,
        "Galat ResNet: sample SD 3 seed, bukan CI sesi. BEATs satu model tetap.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(output / "baseline-vs-nested.png", dpi=180)
    fig.savefig(output / "baseline-vs-nested.pdf")
    plt.close(fig)

    real = [t for t in selection["tasks"] if not t["permuted"]]
    maximum = max(5, np.ceil(max(max(t["inner_accuracy"]) for t in real) * 100))
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), layout="constrained")
    for axis, family, title in zip(
        axes, ("beats", "resnet"), ("BEATs", "ResNet per seed"), strict=True
    ):
        tasks = [t for t in real if (t["encoder"] == "beats") == (family == "beats")]
        values = np.asarray([t["inner_accuracy"] for t in tasks]) * 100
        picture = axis.imshow(values, aspect="auto", cmap="viridis", vmin=0, vmax=maximum)
        axis.scatter(
            [t["selected_index"] for t in tasks],
            np.arange(len(tasks)),
            facecolor="white",
            edgecolor="black",
            s=35,
            linewidth=0.7,
        )
        axis.set_yticks(
            np.arange(len(tasks)),
            [
                t["heldout"] if family == "beats" else f"{t['heldout']}:s{t['encoder'][-2:]}"
                for t in tasks
            ],
            fontsize=8,
        )
        axis.set_xticks(np.arange(len(ALPHAS)), [f"{a:g}" for a in ALPHAS], rotation=50)
        axis.set_title(title)
        axis.set_xlabel("α Ridge (titik putih = pilihan terkunci)")
        axis.set_ylabel("Folder uji luar / seed; nilai berasal dari 5 folder latih")
    fig.colorbar(picture, ax=axes, label="Akurasi validasi internal (%)", shrink=0.7)
    fig.suptitle("Kurva seleksi internal: setiap baris menahan folder uji luar", fontsize=12)
    fig.savefig(output / "validasi-internal.png", dpi=160)
    fig.savefig(output / "validasi-internal.pdf")
    plt.close(fig)


def run(root, audit_dir, output, snapshot):
    result = read_json(root / "results.json")
    selection = read_json(root / "selections.json")
    audit = read_json(audit_dir / "audit.json")
    if (
        audit["status"] != "passed"
        or audit["run_results_sha256"] != file_hash(root / "results.json")
        or audit["selection_lock_sha256"] != file_hash(root / "selections-locked.json")
    ):
        raise ResearchError("Audit tidak cocok dengan hasil/lock seleksi.")
    for folder, record in ((root, result), (audit_dir, audit)):
        for name, digest in record["artifact_sha256"].items():
            if file_hash(folder / name) != digest:
                raise ResearchError(f"Artefak berubah: {folder / name}")
    analysis = analyse(root, result)
    if snapshot and snapshot.exists():
        raise ResearchError("Snapshot sudah ada.")
    fresh_directory(output)
    plots(analysis, selection, output)
    condition_rows = []
    for row in result["conditions"]:
        entry = {
            k: row[k]
            for k in (
                "id",
                "heldout",
                "encoder",
                "permuted",
                "selected_alpha",
                "inner_selected_accuracy",
                "support_accuracy",
                "difference_accuracy",
            )
        }
        for method, key in (("baseline", "baseline_test"), ("nested", "test")):
            entry.update(
                {
                    f"{method}_{name}": row[key][name]
                    for name in ("n", "correct", "accuracy", "top5", "macro_f1")
                }
            )
        condition_rows.append(entry)
    curves = []
    splits = {s["heldout"]: s for s in read_json(root / "split-plan.json")["outer_splits"]}
    for task in selection["tasks"]:
        for index, alpha in enumerate(ALPHAS):
            for fold, split in enumerate(splits[task["heldout"]]["inner"]):
                correct = task["inner_correct"][index][fold]
                curves.append(
                    {
                        "task": task["id"],
                        "heldout": task["heldout"],
                        "encoder": task["encoder"],
                        "permuted": task["permuted"],
                        "alpha": alpha,
                        "validation_folder": split["validation_folder"],
                        "n": 216,
                        "correct": correct,
                        "accuracy": correct / 216,
                        "support_accuracy": task["inner_support_accuracy"][index][fold],
                    }
                )
    for name, rows in (("per-condition.csv", condition_rows), ("inner-validation.csv", curves)):
        with (output / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    normal = {r["family"]: r for r in analysis["comparisons"] if not r["permuted"]}
    negative = {r["family"]: r for r in analysis["comparisons"] if r["permuted"]}
    rows = [r for r in result["conditions"] if not r["permuted"]]
    support = {
        family: float(np.mean([r["support_accuracy"] for r in rows if r["family"] == family]))
        for family in ("beats", "resnet")
    }
    selected_inner = {
        family: float(
            np.mean([r["inner_selected_accuracy"] for r in rows if r["family"] == family])
        )
        for family in ("beats", "resnet")
    }
    choices = next(r for r in analysis["alpha_choices"] if not r["permuted"])
    cpu_change_count = len(audit["cpu_feature_prediction_changes"])
    lines = [
        "# Regularisasi bersarang MKA: hasil 8 November 2026",
        "",
        "**Pemilihan alpha dari folder latih belum memperbaiki transfer.** "
        f"BEATs berubah dari {display(normal['beats']['baseline']['test']['accuracy'])} "
        f"menjadi {display(normal['beats']['nested']['test']['accuracy'])}; "
        f"ResNet dari {display(normal['resnet']['baseline']['test']['accuracy'])} "
        f"menjadi {display(normal['resnet']['nested']['test']['accuracy'])}. "
        "Peluang uniform 36 kelas adalah 2,78%. Ini deskripsi hasil, "
        "bukan uji kesetaraan statistik.",
        "",
        "Enam outer fold memakai lima folder untuk fit/seleksi dan satu folder "
        "untuk uji. Lima inner fold memisahkan empat folder fit (864 potongan, "
        "144 WAV) dan satu validasi (216 potongan, 36 WAV) di dalam outer-train. "
        "Tidak ada WAV induk/PCM potongan melintasi ketiga peran. Semua tugas "
        "memakai K6: 1.080 label outer-train, 216 potongan outer-test. "
        "Embedding/crop sama dengan studi sebelumnya dan tidak dilatih ulang.",
        "",
        "Grid tetap 1e-6/1e-5/1e-4/1e-3/1e-2/0,1/1/10/100. "
        "Pemenang memaksimalkan jumlah benar dari lima inner validation, "
        "seri memilih alpha terbesar. Semua 48 pilihan dikunci sebelum "
        "prediksi luar baru pertama. Baseline alpha1 dipakai ulang dari studi "
        "LOPO yang sudah diaudit. Tidak menghitung outer score untuk alpha "
        "lain atau memilih alpha melalui folder uji.",
        "",
        "## Seluruh folder uji",
        "",
        "| Folder | BEATs α1 | BEATs nested | ResNet α1 | ResNet nested |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in analysis["per_folder"]:
        if row["permuted"]:
            continue
        lines.append(
            f"| {row['heldout']} | {display(row['beats']['baseline_accuracy'])} | "
            f"{display(row['beats']['nested_accuracy'])} | "
            f"{display(row['resnet']['baseline_accuracy'])} | "
            f"{display(row['resnet']['nested_accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "Gabungan out-of-fold mencakup 1.296 potongan unik, setiap potongan "
            "menjadi uji sekali per konfigurasi. ResNet nested per seed42/43/44: "
            + ", ".join(
                percent(v) + "%" for v in normal["resnet"]["nested"]["test"]["accuracy"]["values"]
            )
            + ". Sample SD mengukur tiga inisialisasi, bukan sesi. BEATs satu model "
            "tetap, tanpa ulangan semu atau ensemble. Fold berbagi data latih.",
            "",
            "![Baseline dan nested](baseline-vs-nested.png)",
            "",
            "## Pilihan alpha dan kemampuan fit",
            "",
            f"Dari 24 tugas normal, {choices['lower_grid_edge']} memilih batas bawah "
            f"dan {choices['upper_grid_edge']} memilih batas atas. Grid tidak diperluas "
            "setelah hasil dibaca. Nilai terpilih bervariasi antar folder/seed, "
            "tanpa satu alpha yang konsisten untuk semua predictor.",
            "",
            "| Folder uji | Alpha BEATs | Alpha ResNet s42 / s43 / s44 |",
            "|---|---:|---|",
        ]
    )
    for row in analysis["per_folder"]:
        if not row["permuted"]:
            lines.append(
                f"| {row['heldout']} | {row['beats']['selected_alphas'][0]:g} | "
                + " / ".join(f"{a:g}" for a in row["resnet"]["selected_alphas"])
                + " |"
            )
    lines.extend(
        [
            "",
            "![Kurva validasi internal](validasi-internal.png)",
            "",
            f"Rerata dukungan setelah refit: BEATs {percent(support['beats'])}%, "
            f"ResNet {percent(support['resnet'])}%; baseline alpha1 sebelumnya "
            "70,32% dan 29,67%. BEATs mencapai dukungan 100% pada hp, MSI, Mac "
            "dan Zoom, tetapi outer accuracy keempatnya 0,00–2,78%. "
            "Peningkatan fit belum berubah menjadi transfer.",
            "",
            f"Rerata validasi internal pada pemenang BEATs "
            f"{percent(selected_inner['beats'])}%, ResNet {percent(selected_inner['resnet'])}%. "
            "Angka ini dipakai untuk seleksi sehingga optimis dan tidak boleh "
            "dijadikan perkiraan akurasi final yang bebas bias.",
            "",
            "## Perbandingan berpasangan dan kontrol",
            "",
            "BEATs memperbaiki 18 contoh yang sebelumnya salah, tetapi membuat "
            "23 kesalahan baru: benar total turun dari 28 menjadi 23/1.296. "
            "Pada tiga ResNet, 78 prediksi menjadi benar dan 78 menjadi salah; "
            "rerata akurasi sama meskipun prediksi berubah. Hitungan ResNet ini "
            "menjumlahkan tiga model pada contoh yang sama, bukan 3.888 audio unik.",
            "",
            f"Kontrol negatif memakai permutasi label tingkat WAV seed997 yang "
            f"sama dengan baseline, termasuk label inner validation yang konsisten. "
            f"Nested BEATs {display(negative['beats']['nested']['test']['accuracy'])}; "
            f"ResNet {display(negative['resnet']['nested']['test']['accuracy'])}. "
            "Ini satu kontrol tetap, bukan distribusi null atau uji signifikansi.",
            "",
            "Agregasi enam skor potongan per WAV juga rendah: nested BEATs "
            f"{display(normal['beats']['nested']['parent']['accuracy'])}, ResNet "
            f"{display(normal['resnet']['nested']['parent']['accuracy'])} "
            "pada 216 WAV out-of-fold. Ini klasifikasi kelompok, bukan decoding teks.",
            "",
            "## Audit dan biaya",
            "",
            f"Semua {audit['inner_heads_replayed']:,} fit inner, "
            f"{audit['outer_heads_replayed']} refit akhir dan 48 baseline alpha1 "
            "diulang. Persamaan ridge terpusat juga diselesaikan secara langsung; "
            "koefisien/skor memenuhi toleransi 1e-7, prediksi/metrik dan semua "
            "pilihan alpha identik. Catatan waktu dan hash menunjukkan lock "
            "seleksi mendahului evaluasi baru.",
            "",
            f"Selisih maksimum koefisien inner {audit['inner_maximum_coef_difference']:.3g}, "
            f"skor {audit['inner_maximum_score_difference']:.3g}. "
            f"Embedding CPU digunakan kembali dari audit LOPO yang dipatok, "
            "tanpa mengulang ekstraksi encoder. "
            + (
                "Tidak ada perubahan prediksi saat head memakai embedding CPU."
                if not audit["cpu_feature_prediction_changes"]
                else f"Ada perubahan prediksi numerik pada {cpu_change_count} tugas; "
                "rincian dan akurasi CPU ada di audit.json."
            ),
            "",
            f"Seleksi/penyimpanan {selection['seconds']:.2f} detik, "
            f"refit/evaluasi/verifikasi akhir {result['seconds']:.2f} detik, "
            f"audit {audit['seconds']:.2f} detik. Waktu load/hash awal tidak "
            "dimasukkan dalam seleksi. Semua tahap CPU; tidak menggunakan GPU "
            "atau memperbarui backbone. Tersedia 2.160 baris validasi dan "
            f"{audit['prediction_rows']:,} prediksi potongan/WAV akhir untuk diperiksa.",
            "",
            "## Batas dan langkah berikut",
            "",
            "Pada embedding beku dan grid ini, alpha1 tetap bukan penjelasan "
            "yang cukup untuk kegagalan transfer: seleksi regularisasi yang "
            "memisahkan folder uji belum memperbaikinya. Ini tidak membuktikan "
            "seluruh representasi/optimasi/segmentasi mustahil berhasil atau "
            "bahwa sedikit label target pasti akan memperbaiki hasil.",
            "",
            "Studi dipicu hasil uji sebelumnya; seluruh folder sudah dianalisis "
            "dan label menjadi fit di fold lain. Nested CV menjaga pemilihan "
            "alpha setiap predictor, tetapi tidak membuat dataset kembali buta. "
            "Nama folder bukan ID perangkat/sesi fisik. Pemisahan WAV/PCM belum "
            "menolak semua salinan transformasi. Perubahan kanal, keyboard, "
            "cara mengetik dan segmentasi belum terpisah secara kausal.",
            "",
            "Langkah yang lebih informatif berikutnya adalah memeriksa manfaat "
            "label domain target dengan data kalibrasi/uji dari rekaman terpisah. "
            "MKA masih hanya menyediakan satu WAV per tombol/folder sehingga "
            "belum mendukung pertanyaan efisiensi kalibrasi ke sesi baru itu. "
            "Kontrol dalam WAV dapat menjawab pertanyaan diagnostik berbeda, "
            "dengan protokol terpisah dan tanpa klaim generalisasi sesi baru. "
            "Tidak memperluas grid atau mengganti model dari hasil uji ini.",
            "",
            "## Artefak",
            "",
            "- [Protokol](../../research/MKA_NESTED_PROTOCOL.md), "
            "[lock pilihan alpha](../../runs/mka-nested/selections-locked.json).",
            "- [48 kondisi luar](per-condition.csv), [2.160 validasi](inner-validation.csv), "
            "[ringkasan JSON](summary.json).",
            "- [Koefisien/skor inner](../../runs/mka-nested/inner/) dan "
            "[head akhir](../../runs/mka-nested/conditions/).",
            "- [Audit](../mka-nested-audit/audit.json), "
            "[prediksi berpasangan](../mka-nested-audit/classifications.csv).",
            "- [Gambar utama PDF](baseline-vs-nested.pdf), "
            "[kurva internal PDF](validasi-internal.pdf).",
            "",
        ]
    )
    (output / "laporan.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "created_at": now(),
        "status": "complete",
        "analysis": analysis,
        "mean_selected_support_accuracy": support,
        "mean_selected_inner_accuracy": selected_inner,
        "run_results_sha256": file_hash(root / "results.json"),
        "selection_lock_sha256": file_hash(root / "selections-locked.json"),
        "audit_sha256": file_hash(audit_dir / "audit.json"),
        "report_code_sha256": file_hash(Path(__file__)),
        "analysis_code_sha256": file_hash(ROOT / "src/ketik_audio/mka_nested_analysis.py"),
        "artifact_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir())},
    }
    write_json(output / "summary.json", summary)
    if snapshot:
        write_json(snapshot, summary)
    print(f"Reported all nested tasks: {output / 'laporan.md'}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/mka-nested")
    parser.add_argument("--audit", type=Path, default=ROOT / "reports/mka-nested-audit")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/mka-nested")
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    try:
        run(args.run, args.audit, args.output, args.snapshot)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
