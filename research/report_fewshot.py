"""Render the locked few-shot pilot; refuse incomplete or inconsistent results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ketik_audio.common import ResearchError, file_hash, fresh_directory, read_json, write_json
from ketik_audio.fewshot import KS, SUPPORT_SEEDS

NAMES = {
    "coatnet_scratch": "CoAtNet scratch",
    "coatnet_adapt": "CoAtNet adaptasi",
    "beats_linear": "BEATs + linear",
}
COLORS = {"coatnet_scratch": "#4c78a8", "coatnet_adapt": "#e17c24", "beats_linear": "#15967d"}


def group_results(conditions):
    groups = defaultdict(list)
    identities = set()
    for row in conditions:
        identity = (row["source"], row["target"], row["method"], row["k"], row["support_seed"])
        if identity in identities:
            raise ResearchError("Kondisi terduplikasi.")
        identities.add(identity)
        if row["test"]["n"] != 360 or row["target_labels_used"] != 36 * row["k"]:
            raise ResearchError("Jumlah contoh uji/label melanggar protokol.")
        groups[(row["source"], row["target"], row["method"], row["k"])].append(row)
    expected = {
        (source, target, method, k, seed)
        for source, target in [("phone", "zoom"), ("zoom", "phone")]
        for method in NAMES
        for k in KS
        for seed in SUPPORT_SEEDS
    } | {
        (source, target, method, 0, None)
        for source, target in [("phone", "zoom"), ("zoom", "phone")]
        for method in ["coatnet_adapt", "beats_linear"]
    }
    if identities != expected:
        raise ResearchError("Eksperimen belum mencakup tepat 94 kondisi yang ditetapkan.")
    summaries = []
    for (source, target, method, k), rows in sorted(groups.items()):
        summary = {
            "source": source,
            "target": target,
            "method": method,
            "k": k,
            "target_labels": k * 36,
            "draws": len(rows),
            "test_events_reused": 360,
        }
        for field in ["accuracy", "macro_f1", "top5"]:
            values = [r["test"][field] for r in rows]
            summary[f"{field}_mean"] = float(np.mean(values))
            summary[f"{field}_sd"] = float(np.std(values, ddof=1)) if len(rows) > 1 else None
        summary["correct_counts"] = [r["test"]["correct"] for r in rows]
        for field in ["training_seconds", "inference_seconds"]:
            summary[f"{field}_mean"] = float(np.mean([r[field] for r in rows]))
        support_accuracy = [r["support_accuracy"] for r in rows if "support_accuracy" in r]
        summary["support_accuracy_mean"] = (
            float(np.mean(support_accuracy)) if support_accuracy else None
        )
        summaries.append(summary)
    deltas = []
    for source, target in [("phone", "zoom"), ("zoom", "phone")]:
        for k in KS:
            by_method = {
                method: {r["support_seed"]: r for r in groups[source, target, method, k]}
                for method in NAMES
            }
            values = [
                by_method["coatnet_adapt"][s]["test"]["accuracy"]
                - by_method["coatnet_scratch"][s]["test"]["accuracy"]
                for s in SUPPORT_SEEDS
            ]
            deltas.append(
                {
                    "source": source,
                    "target": target,
                    "k": k,
                    "paired_accuracy_delta_mean": float(np.mean(values)),
                    "paired_accuracy_delta_sd": float(np.std(values, ddof=1)),
                    "per_seed": dict(zip(map(str, SUPPORT_SEEDS), values, strict=True)),
                }
            )
    return summaries, deltas


def cell(value, sd):
    return f"{value * 100:.2f}" + (f" ± {sd * 100:.2f}" if sd is not None else "")


def save_figure(figure, output, name):
    figure.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    figure.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def figures(summaries, deltas, root, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for metric, label in [("accuracy", "Top-1 (%)"), ("macro_f1", "Macro-F1 (%)")]:
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for axis, target in zip(axes, ["zoom", "phone"], strict=True):
            for method, name in NAMES.items():
                rows = sorted(
                    [s for s in summaries if s["target"] == target and s["method"] == method],
                    key=lambda s: s["k"],
                )
                positive = [s for s in rows if s["k"]]
                axis.errorbar(
                    [s["k"] for s in positive],
                    [s[f"{metric}_mean"] * 100 for s in positive],
                    yerr=[s[f"{metric}_sd"] * 100 for s in positive],
                    color=COLORS[method],
                    label=name,
                    marker="o",
                    capsize=3,
                )
                zero = [s for s in rows if not s["k"]]
                if zero:
                    axis.scatter(
                        0, zero[0][f"{metric}_mean"] * 100, color=COLORS[method], marker="D", s=35
                    )
            axis.axhline(100 / 36, color="#aaa", linestyle=":", linewidth=1)
            axis.axhline(80, color="#bbb", linestyle="--", linewidth=0.8)
            axis.axhline(90, color="#bbb", linestyle="--", linewidth=0.8)
            axis.set_xticks([0, *KS])
            axis.set_ylim(0, 103)
            axis.set_title("Ponsel → Zoom" if target == "zoom" else "Zoom → ponsel")
            axis.set_xlabel("Contoh kalibrasi per tombol; total label = 36K")
        axes[0].set_ylabel(label)
        axes[1].legend(loc="upper left", fontsize=9)
        figure.suptitle(
            "Harrison: CoAtNet target 1000 update; BEATs encoder beku + linear\n"
            "Rerata ± SD tiga dukungan; 360 event uji per kanal; K=0 satu evaluasi; eksploratif",
            fontsize=11,
        )
        figure.tight_layout()
        save_figure(figure, output, f"kurva-{metric}")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, target in zip(axes, ["zoom", "phone"], strict=True):
        rows = sorted([d for d in deltas if d["target"] == target], key=lambda d: d["k"])
        axis.errorbar(
            [r["k"] for r in rows],
            [r["paired_accuracy_delta_mean"] * 100 for r in rows],
            yerr=[r["paired_accuracy_delta_sd"] * 100 for r in rows],
            marker="o",
            capsize=4,
            color=COLORS["coatnet_adapt"],
        )
        axis.axhline(0, color="#666", linestyle="--")
        axis.set_xticks(KS)
        axis.set_xlabel("Contoh kalibrasi per tombol")
        axis.set_title("Ponsel → Zoom" if target == "zoom" else "Zoom → ponsel")
    axes[0].set_ylabel("Adaptasi − scratch (poin persentase)")
    figure.suptitle("Selisih dipasangkan pada dukungan yang sama; rerata ± SD, n=3 draw")
    figure.tight_layout()
    save_figure(figure, output, "selisih-adaptasi")

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
    for col, domain in enumerate(["phone", "zoom"]):
        history = read_json(root / f"source-{domain}/history.json")["epochs"]
        source = read_json(root / f"source-{domain}/result.json")
        axes[0, col].plot([r["epoch"] for r in history], [r["loss"] for r in history])
        valid = [r for r in history if "validation_accuracy" in r]
        axes[1, col].plot(
            [r["epoch"] for r in valid], [r["validation_accuracy"] * 100 for r in valid]
        )
        axes[1, col].axvline(source["best_epoch"], color="#666", linestyle="--")
        axes[0, col].set_title(domain)
        axes[1, col].set_xlabel("Epoch sumber")
        axes[1, col].set_ylim(0, 101)
    axes[0, 0].set_ylabel("Cross-entropy augmented train")
    axes[1, 0].set_ylabel("Top-1 validasi sumber (%)")
    figure.suptitle("Pretraining ulang CoAtNet pada frontend 16 kHz yang seragam")
    figure.tight_layout()
    save_figure(figure, output, "kurva-sumber")


def report(root: Path, output: Path, audit: Path | None):
    record = read_json(root / "results.json")
    manifest = read_json(root / "manifest.json")
    if record["status"] != "complete" or manifest["status"] != "complete":
        raise ResearchError("Tunggu semua kondisi selesai sebelum membuat laporan akhir.")
    if record["signature"] != manifest["signature"]:
        raise ResearchError("Manifest dan hasil berbeda.")
    if file_hash(root / "protocol.md") != record["signature"]["protocol_sha256"]:
        raise ResearchError("Salinan protokol run berubah.")
    summaries, deltas = group_results(record["conditions"])
    audit_result = read_json(audit) if audit else None
    if audit_result and audit_result["source_results_sha256"] != file_hash(root / "results.json"):
        raise ResearchError("Audit berasal dari hasil lain.")
    fresh_directory(output)
    figures(summaries, deltas, root, output)
    write_json(
        output / "summary.json",
        {
            "results_sha256": file_hash(root / "results.json"),
            "signature": record["signature"],
            "groups": summaries,
            "paired_deltas": deltas,
        },
    )
    with (output / "ringkasan.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    rows = []
    for condition in record["conditions"]:
        row = {
            k: condition[k]
            for k in [
                "source",
                "target",
                "method",
                "k",
                "support_seed",
                "target_labels_used",
                "training_seconds",
                "inference_seconds",
            ]
        }
        row.update(
            {k: condition["test"][k] for k in ["correct", "n", "accuracy", "macro_f1", "top5"]}
        )
        row["support_accuracy"] = condition.get("support_accuracy")
        row["peak_gpu_allocated_bytes"] = condition.get("peak_gpu_allocated_bytes")
        rows.append(row)
    with (output / "semua-percobaan.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Pilot: berapa contoh per tombol untuk kalibrasi kanal?",
        "",
        "Hasil eksploratif pada rekaman eksternal Harrison, bukan holdout konfirmatori baru. "
        "Tiga pilihan dukungan memakai 360 event uji yang sama per kanal; SD menggambarkan "
        "pilihan dukungan, bukan variasi sesi atau populasi. Tidak ada tuning pada label uji.",
        "",
        "**Batas perbandingan:** CoAtNet scratch pada 1000 update belum menguasai "
        "dukungannya sendiri. Keunggulan pada tabel tidak membuktikan efisiensi label "
        "dibanding scratch yang telah dilatih memadai. "
        "[Diagnostik optimisasi terpisah](../fewshot-diagnostic/laporan.md) memeriksa "
        "scratch pada 10.000 update, K=10 dan satu seed; tidak menggantikan hasil utama.",
        "",
        "![Kurva top-1](kurva-accuracy.png)",
        "",
        "## Akurasi top-1",
        "",
        "Nilai dalam persen: rerata ± SD sampel, tiga draw. "
        "K=0 hanya sekali. K adalah jumlah label **per tombol**, anggaran 36K per fit. "
        "Pilot keseluruhan memakai kumpulan 15 kandidat kalibrasi dan 10 label uji "
        "per tombol; anggaran per fit tidak menggambarkan seluruh biaya anotasi eksperimen.",
        "",
    ]
    for source, target in [("phone", "zoom"), ("zoom", "phone")]:
        lines += [
            f"### {source} → {target}",
            "",
            "| K | Label target | CoAtNet scratch | CoAtNet adaptasi | BEATs + linear |",
            "|---:|---:|---:|---:|---:|",
        ]
        for k in [0, *KS]:
            entries = {r["method"]: r for r in summaries if r["target"] == target and r["k"] == k}
            values = [
                cell(entries[m]["accuracy_mean"], entries[m]["accuracy_sd"])
                if m in entries
                else "—"
                for m in NAMES
            ]
            lines.append(f"| {k} | {36 * k} | " + " | ".join(values) + " |")
        lines += [""]
    lines += [
        "## Ambang yang tercapai pada grid pilot",
        "",
        "K pertama dengan **rerata** top-1 melampaui ambang berikut. Ini bukan minimum "
        "populasi, dan tiga draw tidak memberi jaminan untuk sesi baru.",
        "",
        "| Target | Metode | ≥80% | ≥90% |",
        "|---|---|---:|---:|",
    ]
    for target in ["zoom", "phone"]:
        for method, name in NAMES.items():
            entries = sorted(
                [
                    r
                    for r in summaries
                    if r["target"] == target and r["method"] == method and r["k"]
                ],
                key=lambda r: r["k"],
            )
            values = [
                next(
                    (str(r["k"]) for r in entries if r["accuracy_mean"] >= threshold),
                    "belum sampai K=10",
                )
                for threshold in [0.8, 0.9]
            ]
            lines.append(f"| {target} | {name} | {values[0]} | {values[1]} |")
    lines += [
        "",
        "## Perbandingan adaptasi dan scratch pada anggaran update ini",
        "",
        "![Selisih adaptasi dan scratch](selisih-adaptasi.png)",
        "",
        "| Arah | K | Adaptasi − scratch (pp), rerata ± SD |",
        "|---|---:|---:|",
    ]
    for d in deltas:
        lines.append(
            f"| {d['source']} → {d['target']} | {d['k']} | "
            f"{cell(d['paired_accuracy_delta_mean'], d['paired_accuracy_delta_sd'])} |"
        )
    lines += [
        "",
        "Kedua CoAtNet menggunakan dukungan dan normalisasi dukungan yang identik, "
        "tetapi learning rate adaptasi 1e-4 dan scratch 5e-4. Masing-masing tepat 1000 "
        "update. Selisih tersebut tidak mengisolasi inisialisasi bobot dari learning rate "
        "atau menjamin kedua model sudah konvergen.",
        "",
        "## Kualitas model sumber",
        "",
        "| Kanal sumber | Best epoch / 600 | Validasi | Uji sumber | Waktu latih |",
        "|---|---:|---:|---:|---:|",
    ]
    for domain, r in record["sources"].items():
        lines.append(
            f"| {domain} | {r['best_epoch']} | {r['validation_accuracy'] * 100:.2f}% | "
            f"{r['source_test']['accuracy'] * 100:.2f}% "
            f"({r['source_test']['correct']}/{r['source_test']['n']}) | "
            f"{r['training_seconds']:.1f} s |"
        )
    lines += [
        "",
        "![Pelatihan sumber](kurva-sumber.png)",
        "",
        "Setiap sumber memakai 648 label train dan 72 label validasi; 180 contoh uji "
        "sumber tidak dipakai memilih checkpoint. Pretraining memakai 600 epoch dan "
        "frontend baru; angka ini tidak dapat dijadikan ablation tunggal terhadap "
        "reproduksi 1100 epoch dengan durasi dan sample rate lama.",
        "",
        "## Waktu, memori, dan data pralatih",
        "",
        f"Perangkat neural terverifikasi: **{manifest['hardware']['name']}**, "
        f"backend {manifest['hardware']['backend']}. Model CoAtNet sekitar 1,97 juta "
        f"parameter; BEATs {manifest['cache']['parameters']:,} parameter dibekukan.",
        "",
    ]
    cache = manifest["cache"]
    times = cache["extraction_seconds_including_mel"]
    lines += [
        f"Ekstraksi fitur seluruh 1800 klip sekali (termasuk mel CoAtNet): "
        f"{sum(times.values()):.2f} s; phone {times['phone']:.2f} s, Zoom {times['zoom']:.2f} s. "
        f"Puncak alokasi GPU saat ekstraksi {cache['peak_gpu_allocated_bytes'] / 2**20:.1f} MiB. "
        "Waktu tersebut mencakup startup kernel awal dan memakai batch 16; bukan "
        "pengukuran latensi satu ketukan.",
        "",
        "| Target | Metode, K=10 | Waktu fit rata-rata | Akurasi dukungan rata-rata |",
        "|---|---|---:|---:|",
    ]
    for r in summaries:
        if r["k"] == 10:
            support = (
                f"{r['support_accuracy_mean'] * 100:.2f}%"
                if r["support_accuracy_mean"] is not None
                else "tidak direkam"
            )
            lines.append(
                f"| {r['target']} | {NAMES[r['method']]} | "
                f"{r['training_seconds_mean']:.4f} s | {support} |"
            )
    neural = [r for r in record["conditions"] if r.get("support_accuracy") is not None]
    lines += [
        "",
        f"Akurasi dukungan CoAtNet tanpa augmentasi berkisar "
        f"{min(r['support_accuracy'] for r in neural) * 100:.2f}–"
        f"{max(r['support_accuracy'] for r in neural) * 100:.2f}%. "
        "Nilai rendah menunjukkan keterbatasan optimisasi pada anggaran tetap; "
        "jangan mengaitkan seluruh selisih uji hanya dengan kebutuhan data.",
        "",
        "Fit kepala linear memakai CPU dan fitur yang sudah disimpan; waktu pada tabel "
        "tidak termasuk ekstraksi encoder. CoAtNet memakai GPU; waktu fit tidak termasuk "
        "pretraining sumber. BEATs membawa pengetahuan dari AudioSet AS2M; asal "
        "pretraining berbeda dan ini bukan perbandingan arsitektur murni.",
        "",
        "## Detail protokol dan keterbatasan",
        "",
        "- Potong ulang WAV asli: mono 16 kHz, 400 ms, 50 ms sebelum pusat event. "
        "Resampling polyphase atas rekaman penuh; tanpa padding buatan atau klip berimpit.",
        "- Target event 0–14 per kelas menjadi kumpulan kalibrasi; 15–24 menjadi uji. "
        "Dukungan nested dengan seed 101/102/103. Tidak ada label validasi target.",
        "- CoAtNet memakai log-mel 64×64 dan statistik train/dukungan saja. "
        "K=0 memakai statistik sumber; K>0 menggunakan statistik dukungan. Perubahan "
        "K=0→K>0 mencakup kalibrasi statistik tersebut.",
        "- BEATs memakai Kaldi fbank 38×128; patch menghasilkan 16 token×768, "
        "dirata-ratakan, dinormalisasi L2 per contoh, lalu RidgeClassifier(alpha=1). "
        "Frame ujung yang tidak membentuk patch penuh dibuang oleh encoder aslinya.",
        "- BEATs K=0 memakai kepala dari 648 contoh sumber; K>0 kepala dilatih ulang "
        "hanya pada dukungan target. Encoder tidak diadaptasi.",
        "- Kedua kanal Harrison serta data ujinya telah dipakai pada reproduksi "
        "sebelumnya. Pergantian split tidak menghapus pengetahuan hasil sebelumnya.",
        "- Ketukan satu kelas berasal dari satu WAV. Pemisahan waktu dalam WAV bukan "
        "pemisahan sesi, keyboard, pengguna, atau rekaman independen. Dua arah "
        "menggunakan ulang dua domain yang sama.",
        "- Label phone/J masih tidak pasti karena deteksi count-assisted. Tidak ada "
        "event log untuk mengonfirmasi onset; tidak mengukur deteksi ketukan kontinu.",
        "- Mikrofon, codec/Zoom, sesi dan gaya penekanan berubah bersamaan; penyebab "
        "transfer belum dipisahkan secara kausal. Model besar juga belum tentu lebih "
        "baik pada kalibrasi kecil; nilai klaim berdasarkan hasil di atas.",
        "",
        "## Sumber dan audit",
        "",
        "Data: [Harrison dataset](https://github.com/JBFH-Dev/Keystroke-Datasets), "
        "commit `2ecdc09b6d1868aae50813f6a6c842bdd52dab06`, MIT. "
        "Encoder: [BEATs](https://github.com/microsoft/unilm/tree/master/beats) / "
        "[paper](https://arxiv.org/abs/2212.09058). Kode dan lisensi vendor serta patch "
        "import tersimpan dalam `research/vendor/beats/provenance.json`. Kaldi fbank "
        "memakai implementasi PyTorch murni resmi yang dipatok, tanpa dependensi "
        "ekstensi native TorchAudio yang tidak cocok dengan versi torch di mesin ini.",
        "",
        "Checkpoint asli OneDrive mengembalikan 403 pada pelaksanaan. Salinan "
        "[mirror Hugging Face](https://huggingface.co/lpepino/beats_ckpts/blob/"
        "5b53b0404df452a3a607d7e67687227730e5bad1/BEATs_iter3_plus_AS2M.pt) "
        "dipatok revisinya dan diverifikasi terhadap LFS SHA256 mirror. Ini bukan "
        "digest yang diterbitkan Microsoft.",
        "",
        f"SHA256 hasil lengkap: `{file_hash(root / 'results.json')}`. "
        "`summary.json`, `ringkasan.csv` dan `semua-percobaan.csv` menyimpan nilai terukur. "
        "Setiap direktori run menyimpan indeks, skor, bobot, normalisasi dan konfigurasi.",
    ]
    if audit_result:
        lines += [
            "",
            f"Audit CPU: {audit_result['conditions_replayed']} kondisi diulang dari "
            f"checkpoint/kepala tersimpan; {audit_result['confusion_matches']} confusion "
            "matrix sesuai persis. Pemeriksaan anggaran label, split, nesting dan "
            "normalisasi tercatat dalam audit.json.",
        ]
    lines += [
        "",
        "Langkah konfirmasi: pilih protokol dari hasil eksploratif ini, lalu uji "
        "pada dataset eksternal dengan sesi independen dan hyperparameter yang "
        "sudah dibekukan, sebelum mengumpulkan dataset pribadi.",
        "",
    ]
    (output / "laporan.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Laporan tersimpan: {output / 'laporan.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    try:
        report(args.run, args.output, args.audit)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
