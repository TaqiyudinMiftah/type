"""Write descriptive learning curves and paired channel controls for HEAR and NC1."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from run_public_recording_baseline import AUDITS, RUN

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, read_json, write_json
from ketik_audio.skaid_pilot_common import KS, ROOT, verify

AUDIT = ROOT / "reports/public-recording-baseline-audit"
OUTPUT = ROOT / "reports/public-recording-baseline"
SNAPSHOT = ROOT / "research/results/public-recording-baseline-2026-11-15.json"
NC1_NAMES = {
    "fnatic_quiet": "Fnatic tenang",
    "fnatic_noise": "Fnatic bising",
    "anne_quiet": "Anne Pro tenang",
}
VIEW_NAMES = {"original": "Asli", "mulaw8k": "µ-law"}
PALETTE = ("#2166a5", "#167957", "#975426", "#655281")


def avg(xs):
    if not len(xs):
        raise ResearchError("Agregasi kosong.")
    return float(np.mean(xs))


def pcent(x):
    return f"{100 * x:.2f}".replace(".", ",")


def scoped_rows(trials, scope):
    if scope == "main":
        return [r for r in trials if r["primary"]]
    if scope == "12h":
        return [r for r in trials if r["gap_hours"] >= 12]
    return trials


def hear_summary(trials):
    result = []
    for scope in ("main", "all", "12h"):
        rows = scoped_rows(trials, scope)
        for k, control in [(k, False) for k in (0, *KS)] + [(10, True)]:
            groups = defaultdict(list)
            for r in rows:
                if r["k"] == k and r["permuted"] == control:
                    groups[r["subject"]].append(r)
            people = []
            for person, subset in sorted(groups.items()):
                people.append(
                    {
                        "subject": person,
                        "pairs": len({r["id"] for r in subset}),
                        "trials": len(subset),
                        "accuracy": avg([r["query"]["accuracy"] for r in subset]),
                        "fit_support_accuracy": avg([r["fit_support_accuracy"] for r in subset]),
                        "top3": avg([r["query"]["top3"] for r in subset]),
                    }
                )
            result.append(
                {
                    "scope": scope,
                    "k": k,
                    "permuted": control,
                    "ids": len(people),
                    "pairs": sum(r["pairs"] for r in people),
                    "accuracy": avg([r["accuracy"] for r in people]),
                    "support_fit_accuracy": avg([r["fit_support_accuracy"] for r in people]),
                    "per_id": people,
                }
            )
    return result


def nc1_summary(trials):
    groups = defaultdict(list)
    for r in trials:
        groups[
            (r["id"].split(":")[1], r["calibration_view"], r["query_view"], r["k"], r["permuted"])
        ].append(r)
    rows = []
    for (cond, c, q, k, perm), runs in sorted(groups.items()):
        item = {"condition": cond, "calibration_view": c, "query_view": q, "k": k, "permuted": perm}
        if len(runs) != 3 or any(r["query_indices"] != runs[0]["query_indices"] for r in runs):
            raise ResearchError("Draw atau query NC1 berbeda.")
        rows.append(
            {
                **item,
                "examples": runs[0]["query"]["examples"],
                "classes_present": runs[0]["query"]["classes_present"],
                "accuracy": avg([r["query"]["accuracy"] for r in runs]),
                "balanced_accuracy": avg([r["query"]["balanced_accuracy_present"] for r in runs]),
                "balanced_accuracy_common25": avg(
                    [r["query_common25"]["balanced_accuracy_present"] for r in runs]
                ),
                "support_fit_accuracy": avg([r["fit_support_accuracy"] for r in runs]),
                "top3": avg([r["query"]["top3"] for r in runs]),
            }
        )
    return rows


def paired_codec(trials):
    rows = {
        (r["calibration_view"], r["query_view"], r["k"], r["draw"]): r
        for r in trials
        if r["id"] == "nc1:fnatic_quiet" and not r["permuted"]
    }
    result = []
    for k in KS:
        deltas, penalty, detail = [], [], []
        for d in (101, 102, 103):
            oo, om, mm = (
                rows[a, b, k, d]
                for a, b in (
                    ("original", "original"),
                    ("original", "mulaw8k"),
                    ("mulaw8k", "mulaw8k"),
                )
            )
            if any(oo[f] != x[f] for x in (om, mm) for f in ("support_indices", "query_indices")):
                raise ResearchError("Kontrol kanal tidak memakai sample yang sama.")
            effect = mm["query"]["accuracy"] - om["query"]["accuracy"]
            loss = om["query"]["accuracy"] - oo["query"]["accuracy"]
            deltas.append(effect)
            penalty.append(loss)
            detail.append(
                {"draw": d, "matched_calibration_effect": effect, "codec_effect_unmatched": loss}
            )
        result.append(
            {
                "k": k,
                "matched_calibration_effect": avg(deltas),
                "codec_effect_unmatched": avg(penalty),
                "paired_draws": detail,
            }
        )
    return result


def budget(plan):
    keys = {r["id"]: r["dataset"] for r in plan["records"]}
    return {
        d: {
            "distinct_support_max_k_all_draws": len(
                {
                    i
                    for order in plan["rankings"].values()
                    for rank in order.values()
                    for ids in rank.values()
                    for i in ids
                    if keys[i] == d
                }
            ),
            "distinct_query": len({i for v in plan["queries"].values() for i in v if keys[i] == d}),
            "distinct_support_or_query": sum(v == d for v in keys.values()),
            "sk_aid_recipe_development_labels_shared": 6137,
            "source_fit_labels_k0": 5117 if d == "hear" else None,
        }
        for d in ("hear", "nc1")
    }


def exact(rows, **filters):
    found = [r for r in rows if all(r.get(k) == v for k, v in filters.items())]
    if len(found) != 1:
        raise ResearchError("Baris ringkasan tidak unik.")
    return found[0]


def plot(hear, nc1, output):
    fig, ax = plt.subplots(1, 3, figsize=(14.2, 5.1), layout="constrained")
    for i, (scope, name) in enumerate(
        (("main", "3 pasangan utama"), ("all", "18 pasangan, mean per ID"))
    ):
        values = [exact(hear, scope=scope, k=k, permuted=False)["accuracy"] for k in (0, *KS)]
        ax[0].plot((0, *KS), np.array(values) * 100, marker="o", color=PALETTE[i], label=name)
    ax[0].axhline(100 / 12, ls=":", color="0.55", label="Peluang 12 kelas")
    ax[0].set(title="HEAR: sesi lain, perangkat sama", ylim=(0, 25), xticks=(0, *KS))
    for i, cond in enumerate(NC1_NAMES):
        values = [
            exact(
                nc1,
                condition=cond,
                k=k,
                permuted=False,
                calibration_view="original",
                query_view="original",
            )["accuracy"]
            for k in KS
        ]
        ax[1].plot(KS, np.array(values) * 100, marker="o", color=PALETTE[i], label=NC1_NAMES[cond])
    ax[1].set(title="NC1: kelompok uji 19 Juli", ylim=(0, 55), xticks=KS)
    combos = (
        ("original", "original"),
        ("original", "mulaw8k"),
        ("mulaw8k", "original"),
        ("mulaw8k", "mulaw8k"),
    )
    for i, (a, b) in enumerate(combos):
        values = [
            exact(
                nc1, condition="fnatic_quiet", k=k, permuted=False, calibration_view=a, query_view=b
            )["accuracy"]
            for k in KS
        ]
        ax[2].plot(
            KS,
            np.array(values) * 100,
            marker="o",
            color=PALETTE[i],
            label=f"{VIEW_NAMES[a]} → {VIEW_NAMES[b]}",
        )
    ax[2].set(title="NC1: kontrol codec pada klip sama", ylim=(0, 55), xticks=KS)
    for i, a in enumerate(ax):
        if i:
            a.axhline(100 / 26, ls=":", color="0.55", label="Peluang 26 kelas")
        a.set(xlabel="K contoh per huruf", ylabel="Akurasi top-1 (%)")
        a.grid(axis="y", alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
        a.legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("Baseline MFCC+Ridge terkunci pada rekaman publik terpisah", fontsize=14)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"kurva-rekaman.{ext}", dpi=180)
    plt.close(fig)


def write_report(output, hear, nc1, effects, plan, costs, data, audit):
    lines = [
        "# Baseline pada rekaman publik terpisah",
        "",
        "MFCC+Ridge memakai resep SKAID yang dibekukan. HEAR menguji sesi lain "
        "pada perangkat yang sama menurut metadata. NC1 menguji klip dari tanggal lain "
        "serta pemrosesan µ-law offline. Kedua korpus tidak menguji perubahan mikrofon fisik.",
        "",
        "![Kurva kebutuhan label](kurva-rekaman.png)",
        "",
        "## HEAR: 12 huruf, sesi terpisah",
        "",
        "| K | Label fit target | 3 pasangan utama | Semua 18 pasangan | Gap ≥12 jam |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k in (0, *KS):
        a, b, c = (exact(hear, scope=s, k=k, permuted=False) for s in ("main", "all", "12h"))
        lines.append(
            f"| {k} | {12 * k} | {pcent(a['accuracy'])}% | "
            f"{pcent(b['accuracy'])}% | {pcent(c['accuracy'])}% |"
        )
    lines += [
        "",
        "K0 memakai 5.117 label Phase 1 SKAID; K>0 memasang classifier pada "
        "dukungan HEAR saja. Peluang 8,33%. Rata-rata semua pasangan dibentuk di dalam "
        "masing-masing ID sebelum mean antar ID; gap ≥12 jam hanya dua ID. Tidak ada CI populasi.",
        "",
        "| Pasangan utama | Gap awal sesi (jam) | K10 | Fit dukungan K10 | Kontrol |",
        "|---|---:|---:|---:|---:|",
    ]
    normal = exact(hear, scope="main", k=10, permuted=False)["per_id"]
    control = {r["subject"]: r for r in exact(hear, scope="main", k=10, permuted=True)["per_id"]}
    for row in normal:
        p = exact(plan["paths"], dataset="hear", subject=row["subject"], primary=True)
        lines.append(
            f"| `{p['id']}` | {p['gap_hours']:.2f} | {pcent(row['accuracy'])}% | "
            f"{pcent(row['fit_support_accuracy'])}% | "
            f"{pcent(control[row['subject']]['accuracy'])}% |"
        )
    lines += [
        "",
        "HEAR merilis satu kanal on-device. Sesi diketahui dari metadata, tetapi "
        "awal crop yang cocok dengan PCM induk tidak menjamin onset mekanik tepat. "
        "Setelah audit, tidak ada pasangan yang memenuhi 26 huruf. Hasil yang dekat "
        "peluang tidak menetapkan bahwa semua fitur/model akan gagal.",
        "",
        "## NC1: 26 keluaran, kalibrasi Fnatic tenang 10 Juli",
        "",
        "| K | Label fit | Fnatic tenang | Fnatic bising | Anne Pro tenang |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k in KS:
        rates = [
            pcent(
                exact(
                    nc1,
                    condition=cond,
                    calibration_view="original",
                    query_view="original",
                    k=k,
                    permuted=False,
                )["accuracy"]
            )
            + "%"
            for cond in NC1_NAMES
        ]
        lines.append(f"| {k} | {26 * k} | " + " | ".join(rates) + " |")
    lines += [
        "",
        "Akurasi top-1 menggunakan semua 487/815/795 query yang lolos. Korpus "
        "NC1 didokumentasikan memakai mikrofon yang sama. Quiet 19 Juli tidak memiliki "
        "query `e` yang lolos; classifier masih dapat memprediksi `e`. Noise dan Anne "
        "adalah transfer dari Fnatic tenang, tanpa kalibrasi target pada kondisi itu.",
        "",
        "| Kondisi K10 | Macro recall kelas hadir | Macro recall 25 huruf sama | Kontrol top-1 |",
        "|---|---:|---:|---:|",
    ]
    for cond, name in NC1_NAMES.items():
        filters = dict(condition=cond, calibration_view="original", query_view="original", k=10)
        a, b = (exact(nc1, **filters, permuted=v) for v in (False, True))
        lines.append(
            f"| {name} | {pcent(a['balanced_accuracy'])}% | "
            f"{pcent(a['balanced_accuracy_common25'])}% | {pcent(b['accuracy'])}% |"
        )
    lines += [
        "",
        "## Kontrol kanal pada Fnatic tenang",
        "",
        "| K | Asli→asli | Asli→µ-law | µ-law→asli | µ-law→µ-law | Selisih kalibrasi cocok (pp) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in effects:
        k = r["k"]
        vals = [
            pcent(
                exact(
                    nc1,
                    condition="fnatic_quiet",
                    calibration_view=a,
                    query_view=b,
                    k=k,
                    permuted=False,
                )["accuracy"]
            )
            + "%"
            for a, b in (
                ("original", "original"),
                ("original", "mulaw8k"),
                ("mulaw8k", "original"),
                ("mulaw8k", "mulaw8k"),
            )
        ]
        lines.append(
            f"| {k} | " + " | ".join(vals) + f" | {pcent(r['matched_calibration_effect'])} |"
        )
    lines += [
        "",
        "Selisih dihitung pada query µ-law yang sama, antara kalibrasi µ-law dan asli. "
        "Resampling 16→8→16 kHz dan µ-law dibuat offline; perubahan ini tidak mengukur "
        "sistem VoIP utuh atau efek mikrofon fisik. Ketiga draw memakai ulang ketukan uji.",
        "",
        "## Biaya dan verifikasi",
        "",
        "| Dataset | Union label dukungan K10 | Query unik | Union dukungan/query |",
        "|---|---:|---:|---:|",
        *[
            f"| {d.upper()} | {b['distinct_support_max_k_all_draws']} | {b['distinct_query']} | "
            f"{b['distinct_support_or_query']} |"
            for d, b in costs.items()
        ],
        "",
        "Union HEAR mencakup seluruh 18 pasangan; role suatu sesi dapat berbeda pada "
        "pasangan berbeda, tidak di dalam pasangan yang sama. Label source untuk resep "
        "6.137 termasuk semua 5.117 label fit K0; variasi codec tidak menambah label. "
        "Pool berlabel, pemilihan crop dan label evaluasi di luar label fit classifier.",
        "",
        f"Audit arsip: {data['hear']['wav_decoded']:,} WAV HEAR dan "
        f"{data['nc1']['wav_decoded']:,} WAV NC1 didekode. Audit baseline re-dekode "
        f"{audit['waveforms_redecoded']:,} klip terpilih, menghitung ulang "
        f"{audit['codec_recomputed']:,} versi codec dan replay {audit['trials_replayed']} kondisi "
        f"({audit['heads_replayed']} kepala). Perubahan prediksi replay: "
        f"{sum(audit['prediction_changes'].values())}. "
        "Tidak ada rerata 80% atau 90% pada grid ini.",
        "",
        "Tidak ada pelatihan neural atau penggunaan GPU. Ini klasifikasi potongan, "
        "bukan deteksi audio beruntun. Dataset, urutan sesi dan ukuran kohort berbeda "
        "dari SKAID, sehingga angka tidak dapat dipakai sebagai peringkat SOTA.",
        "",
        "[Protokol](../../research/PUBLIC_RECORDING_BASELINE_PROTOCOL.md) · "
        "[prediksi audit](../public-recording-baseline-audit/classifications.csv) · "
        "[metrik per percobaan](per-trial.csv)",
        "",
    ]
    (output / "laporan.md").write_text("\n".join(lines), encoding="utf-8")


def run(output, snapshot):
    if snapshot.exists():
        raise ResearchError("Snapshot sudah ada.")
    names = {
        RUN: "results.json",
        AUDIT: "audit.json",
        **dict.fromkeys(AUDITS.values(), "summary.json"),
    }
    summaries = {d: read_json(d / n) for d, n in names.items()}
    for d, r in summaries.items():
        verify(d, r)
    sig = {
        str(p.relative_to(ROOT)): file_hash(p)
        for p in [Path(__file__), *(d / n for d, n in names.items())]
    }
    fresh_directory(output)
    plan = read_json(RUN / "plan.json")
    trials = read_json(RUN / "trials.json")["trials"]
    hear = hear_summary([r for r in trials if r["dataset"] == "hear"])
    nc1 = nc1_summary([r for r in trials if r["dataset"] == "nc1"])
    paired = paired_codec(trials)
    costs = budget(plan)
    plot(hear, nc1, output)
    write_report(
        output,
        hear,
        nc1,
        paired,
        plan,
        costs,
        {d: summaries[p] for d, p in AUDITS.items()},
        summaries[AUDIT],
    )
    fields = (
        "id",
        "trial_id",
        "head_id",
        "dataset",
        "subject",
        "primary",
        "k",
        "draw",
        "permuted",
        "calibration_view",
        "query_view",
        "target_fit_labels",
        "source_fit_labels",
        "fit_support_accuracy",
    )
    with (output / "per-trial.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                *fields,
                "examples",
                "accuracy",
                "macro_recall_present",
                "top3",
                "macro_recall_common25",
            ]
        )
        for r in trials:
            q = r["query"]
            common = r["query_common25"]
            w.writerow(
                [
                    *(r[k] for k in fields),
                    q["examples"],
                    q["accuracy"],
                    q["balanced_accuracy_present"],
                    q["top3"],
                    common["balanced_accuracy_present"] if common else None,
                ]
            )
    if any(file_hash(ROOT / p) != h for p, h in sig.items()):
        raise ResearchError("Masukan berubah selama pelaporan.")
    report = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": sig,
        "hear": hear,
        "nc1": nc1,
        "codec_pairs": paired,
        "label_budgets": costs,
        "main_hear_paths": [p for p in plan["paths"] if p["dataset"] == "hear" and p["primary"]],
        "runs": len(trials),
        "thresholds_80_90_reached": any(
            r["accuracy"] >= 0.8 for r in hear + nc1 if not r["permuted"]
        ),
        "artifact_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    write_json(output / "summary.json", report)
    write_json(snapshot, {"report_sha256": file_hash(output / "summary.json"), **report})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    args = parser.parse_args()
    try:
        run(args.output, args.snapshot)
    except (OSError, ResearchError, ValueError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")
