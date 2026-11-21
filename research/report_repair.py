"""Report all source development attempts and the locked phone confirmation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ketik_audio.common import ResearchError, file_hash, fresh_directory, read_json, write_json
from ketik_audio.fewshot import ROOT


def group_summary(variant, rows):
    return {
        "variant": variant,
        "accuracy_mean": float(np.mean([r["test"]["accuracy"] for r in rows])),
        "accuracy_sd": float(np.std([r["test"]["accuracy"] for r in rows], ddof=1)),
        "support_mean": float(np.mean([r["support_accuracy"] for r in rows])),
        "support_sd": float(np.std([r["support_accuracy"] for r in rows], ddof=1)),
        "minimum_support_accuracy": min(r["support_accuracy"] for r in rows),
        "constant_runs": sum(r["support_max_logit_range"] < 1e-4 for r in rows),
        "n_initializations": len(rows),
    }


def run(development, confirmation, audit, old, output, snapshot):
    dev, target, decision = (
        read_json(p)
        for p in [
            development / "results.json",
            confirmation / "results.json",
            development / "decision.json",
        ]
    )
    replay = read_json(audit / "audit.json")
    if (
        replay["status"] != "passed"
        or replay["confirmation_results_sha256"] != file_hash(confirmation / "results.json")
        or replay["development_results_sha256"] != file_hash(development / "results.json")
    ):
        raise ResearchError("Audit hasil terkini belum lulus.")
    previous = read_json(old / "results.json")
    old_audit = read_json(ROOT / "reports/stability-gate/summary.json")
    if old_audit["source_results_sha256"] != file_hash(old / "results.json"):
        raise ResearchError("Audit kontrol lama berbeda dari hasil kontrol.")
    for row in previous["conditions"]:
        old_folder = (
            Path(row["reused_directory"]) if "reused_directory" in row else old / row["directory"]
        )
        checkpoint_record = next(
            r
            for r in old_audit["cpu_replay"]
            if r["variant"] == row["variant"] and r["seed"] == row["training_seed"]
        )
        if file_hash(old_folder / "last.pt") != checkpoint_record["weights_sha256"]:
            raise ResearchError("Checkpoint kontrol lama berubah sejak audit CPU.")
        if "support_max_logit_range" not in row:
            entry = next(
                r
                for r in old_audit["runs"]
                if r["variant"] == row["variant"] and r["seed"] == row["training_seed"]
            )
            row["support_max_logit_range"] = entry["support_max_logit_range"]
    chosen = decision["variant"]
    developments = [(development, dev)]
    if chosen in ("wide-stem-residual", "compact-resnet"):
        parent = ROOT / "runs/repair-development"
        developments.insert(0, (parent, read_json(parent / "results.json")))
    if chosen == "compact-resnet":
        parent = ROOT / "runs/stem-repair-development"
        developments.insert(1, (parent, read_json(parent / "results.json")))
    all_rows = previous["conditions"] + target["conditions"]
    groups = [
        group_summary(variant, [r for r in all_rows if r["variant"] == variant])
        for variant in ("relu", "leaky", chosen)
    ]
    previous_range = {r["variant"]: r for r in groups}
    fresh_directory(output)
    colors = {"relu": "#8d6b8e", "leaky": "#d38336", chosen: "#238c83"}
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    initial = read_json(development / "initial-probes.json")["probes"]
    for j, variant in enumerate(("original", "leaky", chosen)):
        values = [p["input_gradient_l2"] for p in initial if p["variant"] == variant]
        axes[0].scatter(
            np.full(3, j), values, s=35, color=colors["relu" if variant == "original" else variant]
        )
    axes[0].set_yscale("log")
    axes[0].set_xticks(
        range(3), ("ReLU awal", "Leaky awal", "Perbaikan terpilih"), rotation=12, fontsize=8
    )
    axes[0].set_ylabel("Norm gradien input sebelum fit (log)")
    axes[0].set_title("Pemeriksaan mekanisme: Zoom")
    for variant in ("relu", "leaky", chosen):
        histories = []
        for row in sorted(
            [r for r in all_rows if r["variant"] == variant], key=lambda r: r["training_seed"]
        ):
            folder = (
                Path(row["reused_directory"])
                if "reused_directory" in row
                else (confirmation if variant == chosen else old) / row["directory"]
            )
            history = [
                h for h in read_json(folder / "history.json")["updates"] if "support_accuracy" in h
            ]
            histories.append({h["step"]: h["support_accuracy"] for h in history})
        steps = sorted(set.intersection(*(set(h) for h in histories)))
        curves = np.asarray([[h[s] for s in steps] for h in histories]) * 100
        axes[1].plot(steps, curves.mean(0), marker="o", color=colors[variant], label=variant)
        axes[1].fill_between(steps, curves.min(0), curves.max(0), color=colors[variant], alpha=0.12)
        accuracy = [
            r["test"]["accuracy"] * 100
            for r in sorted(
                [r for r in all_rows if r["variant"] == variant], key=lambda r: r["training_seed"]
            )
        ]
        offset = {"relu": -0.25, "leaky": 0, chosen: 0.25}[variant]
        axes[2].bar(
            np.arange(3) + offset, accuracy, width=0.24, color=colors[variant], label=variant
        )
    axes[1].set_xlabel("Update; rerata dan rentang 3 inisialisasi")
    axes[1].set_ylabel("Akurasi dukungan ponsel (%)")
    axes[1].set_ylim(0, 103)
    axes[1].legend(fontsize=8)
    axes[2].set_xticks(range(3), (42, 43, 44))
    axes[2].set_xlabel("Seed inisialisasi; dukungan/uji sama")
    axes[2].set_ylabel("Top-1 uji ponsel (%)")
    axes[2].set_ylim(0, 103)
    axes[2].legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=8, frameon=False
    )
    figure.suptitle(
        "Perbaikan dipilih pada Zoom, lalu dikunci sebelum uji ponsel\n"
        "K=10, 360 label target, 360 event uji, 10.000 update",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(output / "perbaikan.png", dpi=180, bbox_inches="tight")
    figure.savefig(output / "perbaikan.pdf", bbox_inches="tight")
    plt.close(figure)
    lines = [
        "# Perbaikan kestabilan optimisasi",
        "",
        f"Konfigurasi terpilih: **{chosen}**. Pemilihan memakai 360 contoh fit dan "
        "72 validasi **Zoom saja**. Seluruh hasil percobaan pengembangan dicatat "
        "di bawah. Keputusan beserta checksum dikunci sebelum tiga fit ponsel. "
        "Penilaian kelulusan kandidat baru hanya memakai Zoom, tanpa validasi "
        "target. Arah diagnosis berasal dari hasil ponsel historis yang sudah "
        "dilihat; tahap ini tidak membuat ponsel menjadi holdout baru.",
        "",
        "![Gradien dan kestabilan tiga seed](perbaikan.png)",
        "",
        "## Penyebab dan implementasi",
        "",
        "Referensi kecil menginisialisasi proyeksi q/k/v/o attention dengan "
        "normal std=0,001, tanpa jalur residual/LayerNorm pada dua blok attention. "
        "Autopsi sebelumnya menunjukkan variasi antar-input hampir hilang, "
        "bahkan sebelum pelatihan. LeakyReLU pada gate terakhir mencegah satu "
        "jenis kolaps, tetapi belum memulihkan kekuatan sinyal awal.",
        "",
        "Kandidat minimal mengganti inisialisasi delapan matriks attention "
        "menjadi Xavier uniform gain=1, mempertahankan bias nol dan LeakyReLU "
        "terakhir. Kandidat cadangan menambah residual dan pre-LayerNorm pada "
        "attention/MLP token, serta LayerNorm keluaran. Kandidat cadangan hanya "
        "dilatih jika kandidat minimal gagal kriteria pengembangan. Vendor "
        "asli dan eksperimen lama tidak diedit. Setiap fit dimulai dari awal; "
        "perubahan inisialisasi memerlukan retraining.",
        "",
        "Karena kedua kandidat tersebut gagal seleksi pengembangan, protokol "
        "lanjutan menambahkan satu kandidat stem lebar. Stem Conv(1→1) lama "
        "diganti Conv3x3(1→32)–GroupNorm(8)–GELU–Conv3x3(32→64)–"
        "GroupNorm(8)–GELU; ekspansi 1x1 awal dihapus, blok setelah stem tetap "
        "residual/pre-norm. Ini menguji hipotesis bottleneck satu kanal. "
        "Penguasaan dukungan pada varian residual telah membaik, tetapi "
        "validasi seed 43 tetap rendah. Kalibrasi ulang BatchNorm pada salinan "
        "checkpoint sumber tidak memperbaikinya. Hipotesis ini bukan bukti "
        "kontribusi individual stem, normalisasi atau aktivasi.",
        "",
        "Stem lebar juga gagal kriteria: SD validasi 10,79 pp, melewati batas "
        "10 pp. Protokol berikutnya menguji compact-resnet, CNN residual "
        "dengan empat BasicBlock (lebar 16/32/64/128), BatchNorm, ReLU, "
        "shortcut untuk setiap blok dan global average pooling. Konvolusi "
        "Kaiming fan-out, BN terakhir cabang residual diinisialisasi nol. "
        "Model ini lebih kecil dan tidak memakai attention. Ini alternatif "
        "backbone dengan kapasitas berbeda, bukan CoAtNet atau ResNet-18 "
        "kanonis. Rancangan residual merujuk "
        "[He dkk.](https://arxiv.org/abs/1512.03385); efektivitas pada dataset "
        "ini dinilai melalui hasil pengembangan dan konfirmasi di bawah.",
        "",
        "Rujukan implementasi: "
        "[Xavier uniform PyTorch 2.14](https://docs.pytorch.org/docs/2.14/nn.init.html"
        "#torch.nn.init.xavier_uniform_) "
        "dan [residual/pre-norm Transformer](https://github.com/pytorch/pytorch/"
        "blob/main/torch/nn/modules/transformer.py). "
        "Rancangan pre-norm juga merujuk "
        "[Xiong dkk., ICML 2020](https://arxiv.org/abs/2002.04745); hasil paper "
        "tersebut bukan bukti efektivitas pada suara keyboard. "
        "Diagnosis penyusutan sinyal berasal dari kode vendor dan pengukuran lokal, "
        "bukan klaim bahwa seluruh CoAtNet memiliki masalah ini.",
        "",
        "Probe awal menggunakan satu contoh dukungan per kelas (36 total), "
        "mode eval dengan BatchNorm tidak berubah, cross-entropy dan gradien "
        "terhadap mel terstandardisasi. Tidak ada update bobot. Perbandingan "
        "gradien mengukur kondisi awal ini, bukan seluruh lintasan pelatihan.",
        "",
        "## Semua fit pengembangan",
        "",
        "Split sumber Zoom lama: 648 train, 72 validasi, 180 test. Pilih 10 per "
        "kelas dari train (seed 101); 180 test sumber tidak dievaluasi. Fit "
        "6.000 update, Adam 5e-4, batch 16 replacement, LR linear ke nol. "
        "Statistik mel hanya dari 360 dukungan. Kriteria tetap: setiap seed "
        "dukungan >=95%, tidak konstan, rerata validasi >=50%, sample SD "
        "validasi <=10 pp. Ambil kandidat pertama yang lulus, tanpa early stopping.",
        "",
        "| Kandidat | Seed | Dukungan | Validasi | Parameter |",
        "|---|---:|---:|---:|---:|",
    ]
    for dev_folder, record in developments:
        for r in record["conditions"]:
            cfg = read_json(dev_folder / r["directory"] / "config.json")
            lines.append(
                f"| {r['variant']} | {r['training_seed']} | "
                f"{r['support_accuracy'] * 100:.2f}% | "
                f"{r['validation']['accuracy'] * 100:.2f}% | {cfg['parameters']:,} |"
            )
    lines += [
        "",
        f"Kandidat yang dikunci: `{chosen}`. Waktu keputusan: `{decision['created_at']}`.",
        "",
        "## Semua hasil ponsel pada dukungan/uji tetap",
        "",
        "K=10, seed dukungan 101, 360 label target; uji 360 event terakhir. "
        "Tiga inisialisasi, 10.000 update per fit; data, augmentasi, Adam, "
        "LR dan normalisasi sama dengan kontrol lama. Tabel menampilkan "
        "semua seed, tanpa memilih checkpoint atau seed terbaik.",
        "",
        "| Varian | Seed | Dukungan | Top-1 uji | Benar/uji | Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in ("relu", "leaky", chosen):
        for r in sorted(
            [r for r in all_rows if r["variant"] == variant], key=lambda r: r["training_seed"]
        ):
            lines.append(
                f"| {variant} | {r['training_seed']} | "
                f"{r['support_accuracy'] * 100:.2f}% | {r['test']['accuracy'] * 100:.2f}% | "
                f"{r['test']['correct']}/{r['test']['n']} | {r['test']['macro_f1']:.4f} |"
            )
    lines += [
        "",
        "| Varian | Top-1 uji rerata ± SD | Dukungan rerata ± SD | Run konstan |",
        "|---|---:|---:|---:|",
    ]
    for r in groups:
        lines.append(
            f"| {r['variant']} | {r['accuracy_mean'] * 100:.2f}% ± "
            f"{r['accuracy_sd'] * 100:.2f} pp | {r['support_mean'] * 100:.2f}% ± "
            f"{r['support_sd'] * 100:.2f} pp | {r['constant_runs']}/3 |"
        )
    lines += [
        "",
        "SD di atas hanya variasi inisialisasi pada satu pilihan dukungan yang "
        "sama, bukan ketidakpastian populasi atau variasi sesi. Hasil Zoom "
        "pengembangan tidak boleh dibaca sebagai hasil target K=10 tanpa "
        "validasi: 72 label Zoom tambahan dipakai untuk memilih kandidat.",
        "",
        "## Audit dan batas interpretasi",
        "",
        f"Audit CPU lulus untuk **{len(replay['checkpoints'])} checkpoint baru**: "
        "confusion matrix dan metrik cocok, prediksi dukungan/uji individual "
        "cocok, ID contoh/anggaran/normalisasi diperiksa. Hash eksperimen "
        "lama tetap sama. Seluruh fit GPU menggunakan RX 6600 LE.",
        "",
        "Pemisahan masih dalam WAV Harrison yang sama. Uji ponsel telah "
        "dilihat pada tahap riset sebelumnya; ini perbaikan eksploratif, "
        "bukan holdout konfirmatori baru. Ketidakpastian isolasi phone/J "
        "masih berlaku. Keberhasilan optimisasi tidak menyelesaikan "
        "generalisasi ke keyboard, partisipan atau sesi baru. Tiga seed "
        "dan K=10 tidak membuktikan SOTA atau kebutuhan label minimum.",
        "",
        "Protokol: `research/REPAIR_PROTOCOL.md`, `research/STEM_REPAIR_PROTOCOL.md`, "
        "`research/CNN_REPAIR_PROTOCOL.md`. Kode: `src/ketik_audio/repair.py`, "
        "`src/ketik_audio/stem_repair.py`, `src/ketik_audio/cnn_repair.py`. "
        "Audit: `reports/repair-audit/audit.json`.",
        "",
    ]
    (output / "laporan.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "selected_variant": chosen,
        "groups": groups,
        "development": dev,
        "earlier_development": [record for _, record in developments[:-1]],
        "bn_autopsy": read_json(ROOT / "reports/repair-bn-autopsy/results.json"),
        "confirmation": target,
        "decision": decision,
        "cpu_replay": replay["checkpoints"],
        "audit_sha256": file_hash(audit / "audit.json"),
        "test_sd_ratio_to_leaky": previous_range[chosen]["accuracy_sd"]
        / previous_range["leaky"]["accuracy_sd"],
    }
    write_json(output / "summary.json", summary)
    if snapshot:
        if snapshot.exists():
            raise ResearchError("Snapshot sudah ada.")
        write_json(snapshot, summary)
    print(f"Laporan tersimpan: {output / 'laporan.md'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, default=ROOT / "runs/cnn-repair-development")
    parser.add_argument("--confirmation", type=Path, default=ROOT / "runs/cnn-repair-confirmation")
    parser.add_argument("--audit", type=Path, default=ROOT / "reports/repair-audit")
    parser.add_argument("--old", type=Path, default=ROOT / "runs/stability-gate")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/repair")
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    try:
        run(args.development, args.confirmation, args.audit, args.old, args.output, args.snapshot)
    except (ResearchError, OSError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
