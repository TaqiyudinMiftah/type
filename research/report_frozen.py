"""Report frozen-head controls and a support-only autopsy of the failed model."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ketik_audio.common import ResearchError, file_hash, fresh_directory, read_json, write_json
from ketik_audio.fewshot import KS

NAMES = {
    "pretrained_source_norm": "CoAtNet beku, norm sumber",
    "pretrained_support_norm": "CoAtNet beku, norm dukungan",
    "random_source_norm": "CoAtNet acak beku",
    "beats_linear": "BEATs beku + linear",
    "coatnet_adapt": "CoAtNet adaptasi, 1000 update",
}
COLORS = {
    "pretrained_source_norm": "#e18339",
    "pretrained_support_norm": "#b74943",
    "random_source_norm": "#888888",
    "beats_linear": "#15967d",
    "coatnet_adapt": "#4287a8",
}


def aggregate(conditions):
    groups = defaultdict(list)
    for row in conditions:
        groups[(row["target"], row["method"], row["k"])].append(row)
    rows = []
    for (target, method, k), group in sorted(groups.items()):
        row = {
            "target": target,
            "method": method,
            "k": k,
            "draws": len(group),
            "target_labels_per_fit": 36 * k,
        }
        for metric in ["accuracy", "macro_f1", "top5"]:
            values = [r["test"][metric] for r in group]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        row["head_fit_seconds_mean"] = float(np.mean([r["training_seconds"] for r in group]))
        rows.append(row)
    return rows


def cell(row, metric="accuracy"):
    if row is None:
        return "—"
    value, sd = row[f"{metric}_mean"], row[f"{metric}_sd"]
    return f"{value * 100:.2f}" + (f" ± {sd * 100:.2f}" if sd is not None else "")


def run(root: Path, primary: Path, audit: Path, autopsy: Path, output: Path):
    record, previous = read_json(root / "results.json"), read_json(primary / "results.json")
    audited = read_json(audit / "audit.json")
    inspected = read_json(autopsy / "results.json")
    if (
        record["status"] != "complete"
        or audited["conditions_replayed"] != 94
        or audited["source_results_sha256"] != file_hash(root / "results.json")
    ):
        raise ResearchError("Eksperimen/audit belum lengkap atau hash berbeda.")
    fresh_directory(output)
    reference = [
        r for r in previous["conditions"] if r["method"] in ["beats_linear", "coatnet_adapt"]
    ]
    summaries = aggregate(record["conditions"] + reference)
    lookup = {(r["target"], r["method"], r["k"]): r for r in summaries}
    deltas = []
    for target in ["zoom", "phone"]:
        for k in KS:
            kinds = {
                method: {
                    r["support_seed"]: r["test"]["accuracy"]
                    for r in record["conditions"]
                    if r["target"] == target and r["k"] == k and r["method"] == method
                }
                for method in [
                    "pretrained_source_norm",
                    "pretrained_support_norm",
                    "random_source_norm",
                ]
            }
            values = [
                kinds["pretrained_support_norm"][seed] - kinds["pretrained_source_norm"][seed]
                for seed in [101, 102, 103]
            ]
            deltas.append(
                {
                    "target": target,
                    "k": k,
                    "support_minus_source_norm_mean": float(np.mean(values)),
                    "support_minus_source_norm_sd": float(np.std(values, ddof=1)),
                }
            )
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    for metric, label in [("accuracy", "Top-1 (%)"), ("macro_f1", "Macro-F1 (%)")]:
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
        for axis, target in zip(axes, ["zoom", "phone"], strict=True):
            for method, name in NAMES.items():
                rows = [lookup[target, method, k] for k in KS]
                axis.errorbar(
                    KS,
                    [r[f"{metric}_mean"] * 100 for r in rows],
                    yerr=[r[f"{metric}_sd"] * 100 for r in rows],
                    marker="o",
                    capsize=3,
                    color=COLORS[method],
                    label=name,
                    linestyle="--" if method == "coatnet_adapt" else "-",
                )
            axis.set_ylim(0, 102)
            axis.set_xticks(KS)
            axis.set_title("Ponsel → Zoom" if target == "zoom" else "Zoom → ponsel")
            axis.set_xlabel("Label kalibrasi per tombol; 36K label per fit")
            axis.legend(fontsize=7, loc="upper left")
        axes[0].set_ylabel(label)
        figure.suptitle(
            "Kepala Ridge alpha=1, fitur L2; backbone beku termasuk BatchNorm\n"
            "Rerata ± SD tiga dukungan pada 360 event uji tetap; hasil eksploratif",
            fontsize=10,
        )
        figure.tight_layout()
        figure.savefig(output / f"kurva-{metric}.png", dpi=180, bbox_inches="tight")
        figure.savefig(output / f"kurva-{metric}.pdf", bbox_inches="tight")
        plt.close(figure)
    lines = [
        "# Adaptasi kepala klasifikasi dan autopsi optimisasi",
        "",
        "Harrison, 36 kelas, audio 16 kHz/400 ms; RX 6600 LE untuk ekstraksi neural, "
        "CPU untuk Ridge dan audit. Data uji sudah diketahui dari pilot sebelumnya: "
        "ini eksperimen eksploratif, bukan uji sesi baru. Seluruh 94 kondisi baru "
        "diulang dari kepala tersimpan dan cocok persis pada CPU.",
        "",
        "![Kurva top-1](kurva-accuracy.png)",
        "",
        "## Hasil kepala beku",
        "",
        "Rerata ± SD sampel atas seed dukungan 101/102/103, dalam persen. "
        "Dukungan dan test persis sama dengan pilot. Setiap fit memakai 36K label target "
        "dan tidak memakai validasi target. Statistik BatchNorm tidak diadaptasi.",
        "",
    ]
    for target in ["zoom", "phone"]:
        lines += [
            f"### Target {target}",
            "",
            "| K | Pralatih, norm sumber | Pralatih, norm dukungan | "
            "Acak, norm sumber | BEATs + linear |",
            "|---:|---:|---:|---:|---:|",
        ]
        for k in [0, *KS]:
            cells = [
                cell(lookup.get((target, m, k)))
                for m in [
                    "pretrained_source_norm",
                    "pretrained_support_norm",
                    "random_source_norm",
                    "beats_linear",
                ]
            ]
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
        lines += [""]
    lines += [
        "Pada konfigurasi Ridge/L2 yang ditetapkan, pembekuan backbone belum "
        "memperbaiki hasil pilot full fine-tuning: pada K=10, adaptasi 1000 update "
        "sebelumnya mencapai 32,87% di Zoom dan 33,15% di ponsel. Pergantian metode "
        "juga mengubah classifier/loss/optimizer dan penggunaan statistik BN; ini "
        "bukan ablation murni satu faktor pembekuan.",
        "",
        "## Kontrol normalisasi dan representasi",
        "",
        "| Target | K | Norm dukungan − norm sumber (pp), rerata ± SD |",
        "|---|---:|---:|",
    ]
    for row in deltas:
        lines.append(
            f"| {row['target']} | {row['k']} | "
            f"{row['support_minus_source_norm_mean'] * 100:.2f} ± "
            f"{row['support_minus_source_norm_sd'] * 100:.2f} |"
        )
    lines += [
        "",
        "Selisih tersebut memakai encoder dan kepala yang sama pada dukungan "
        "yang dipasangkan; yang berubah adalah statistik normalisasi input. Tidak "
        "memakai label atau statistik uji untuk fitting. Random control memakai "
        "inisialisasi tunggal seed 42 dan berbagi statistik sumber; ia bukan scratch "
        "yang dilatih dan bukan batas kapasitas arsitektur.",
        "",
        "| Sumber | Encoder | Ridge dalam domain sumber (n=180) |",
        "|---|---|---:|",
    ]
    for row in record["source_checks"]:
        lines.append(f"| {row['source']} | {row['kind']} | {row['test']['accuracy'] * 100:.2f}% |")
    lines += [
        "",
        "Representasi pralatih tetap berguna pada domain sumber, tetapi "
        "kepala linear ini belum cukup untuk perpindahan domain. Ini dibatasi "
        "oleh geometri fitur, normalisasi L2 dan regularisasi yang dipilih; bukan "
        "bukti semua informasi tombol hilang. BEATs membawa pretraining AudioSet "
        "berbeda dan jumlah parameter lebih besar, sehingga bukan kontrol arsitektur "
        "murni. Tidak ada metode beku yang mencapai rerata 80–90% sampai K=10.",
        "",
        "## Model ponsel yang menghasilkan keluaran konstan",
        "",
    ]
    failed = next(row for row in inspected["cases"] if row["case"] == "collapsed-phone")
    gate = next(row for row in failed["dead_relu_candidates"] if row["name"] == "0.mlp4.1")
    changed = failed["local_intervention"]
    original_gradient = failed["gradient_probes"][0]["input_gradient_l2"]
    lines += [
        f"Pada 360 contoh dukungan K=10 seed 101, input ReLU `0.mlp4.1` berada "
        f"pada [{gate['input_min']:.8f}, {gate['input_max']:.8f}], seluruhnya negatif. "
        "ReLU menghasilkan nol untuk semua posisi dan contoh. Blok sebelumnya "
        "tidak menerima gradien dari loss; keluaran classifier menjadi identik. "
        "Gradien input juga nol pada salinan mode train, sehingga pergantian "
        "mode BatchNorm saja tidak menjelaskan hambatan langsung ini.",
        "",
        "Intervensi pada salinan model mengganti hanya gate tersebut menjadi "
        "LeakyReLU(0,01), tanpa perubahan bobot atau optimizer step. Pemeriksaan "
        "hanya memakai dukungan, tidak memakai akurasi uji.",
        "",
        "| Pemeriksaan | Asli | Setelah intervensi lokal |",
        "|---|---:|---:|",
        f"| Rentang logit maksimum antardukungan | {failed['max_logit_range']:.6g} | "
        f"{changed['max_logit_range']:.6g} |",
        f"| Norm gradien input, batch seimbang 36 | {original_gradient:.6g} | "
        f"{changed['gradient']['input_gradient_l2']:.6g} |",
        "",
        "Ini menjelaskan jalur yang menghambat pembelajaran pada checkpoint "
        "tersebut. Gradien yang kembali masih kecil; intervensi ini belum membuktikan "
        "perbaikan akurasi atau mengisolasi mengapa bobot awalnya bergerak ke "
        "wilayah negatif. [Pemeriksaan pelatihan lintas inisialisasi]"
        "(../stability-gate/laporan.md) "
        "menguji hipotesis aktivasi secara terpisah. Kode vendor dan hasil lama "
        "tetap utuh.",
        "",
        "## Biaya dan audit",
        "",
        "| Encoder sumber | Jenis | Ekstraksi dua domain | Puncak alokasi GPU |",
        "|---|---|---:|---:|",
    ]
    for row in record["encoders"]:
        lines.append(
            f"| {row['source']} | {row['kind']} | {row['extraction_seconds']:.3f} s | "
            f"{row['peak_gpu_allocated_bytes'] / 2**20:.1f} MiB |"
        )
    extra = sum(r["extra_extraction_seconds"] for r in record["conditions"])
    lines += [
        "",
        f"Ekstraksi tambahan untuk normalisasi dukungan: {extra:.3f} s. Waktu fit "
        "Ridge terpisah dari biaya ekstraksi dan pretraining sumber. Cache fitur "
        "source_norm dipakai ulang; kondisi support_norm memerlukan ekstraksi ulang "
        "karena inputnya berubah. Semua biaya dalam summary.json dan result.json.",
        "",
        "Audit: 94 kepala dan empat pemeriksaan sumber cocok pada CPU. Delapan "
        "subset cache encoder serta 30 subset support_norm juga diulang lewat "
        "encoder CPU. Seluruh parameter dan buffer tetap sama sebelum/sesudah "
        "ekstraksi; statistik sumber/dukungan dan anggaran label diperiksa.",
        "",
        "Protokol: `research/FROZEN_HEAD_PROTOCOL.md`; autopsi lengkap: "
        "`reports/fewshot-autopsy/results.json`; input dan provenance memakai "
        "snapshot pilot. Ketidakpastian phone/J, korelasi dalam WAV, dan penggunaan "
        "ulang data yang sama tetap berlaku. SD tiga dukungan tidak mewakili sesi "
        "atau populasi pengguna baru.",
        "",
        f"SHA256 hasil beku: `{file_hash(root / 'results.json')}`. "
        f"SHA256 autopsi: `{file_hash(autopsy / 'results.json')}`.",
        "",
    ]
    (output / "laporan.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(
        output / "summary.json",
        {
            "groups": summaries,
            "normalization_deltas": deltas,
            "encoders": record["encoders"],
            "source_checks": record["source_checks"],
            "results_sha256": file_hash(root / "results.json"),
            "autopsy_sha256": file_hash(autopsy / "results.json"),
        },
    )
    print(f"Laporan kepala beku dan autopsi tersimpan: {output / 'laporan.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--autopsy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.run, args.primary, args.audit, args.autopsy, args.output)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
