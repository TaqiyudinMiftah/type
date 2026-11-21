"""Compare transfer claims using only the frozen evaluation ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/public-recording-baseline"
AUDIT = ROOT / "reports/public-recording-baseline-audit"
OUTPUT = ROOT / "reports/public-transfer-evidence"
BUDGETS = (1, 2, 3, 5, 10)
DRAWS = (101, 102, 103)
ORIGINAL, CODEC = "original", "mulaw8k"
VIEWS = (ORIGINAL, CODEC)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_ledger():
    audit_path, result_path = AUDIT / "audit.json", RUN / "results.json"
    audit, result = read_json(audit_path), read_json(result_path)
    require(audit["status"] == result["status"] == "complete", "Evaluasi atau audit belum lengkap.")
    trials_path, plan_path = RUN / "trials.json", RUN / "plan.json"
    csv_path = AUDIT / "classifications.csv"
    hear_path = ROOT / "reports/hear-recording-audit/summary.json"
    files = {
        result_path: audit["input_sha256"][str(result_path.relative_to(ROOT))],
        trials_path: result["artifact_sha256"][trials_path.name],
        plan_path: result["artifact_sha256"][plan_path.name],
        csv_path: audit["artifact_sha256"][csv_path.name],
        hear_path: result["input_sha256"][str(hear_path.relative_to(ROOT))],
    }
    for path, expected in files.items():
        require(digest(path) == expected, f"Hash input berbeda: {path.name}")

    trials = read_json(trials_path)["trials"]
    by_id = {t["trial_id"]: t for t in trials}
    require(
        len(by_id) == len(trials) == result["trials"] == audit["trials_replayed"],
        "Jumlah trial berbeda.",
    )
    records = read_json(plan_path)["records"]
    index = {r["id"]: i for i, r in enumerate(records)}
    require(len(index) == len(records), "ID klip tidak unik.")
    correct = defaultdict(dict)
    with csv_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            trial = by_id[row["trial"]]
            i = index[row["clip"]]
            expected = {
                "pair": trial["id"],
                "dataset": trial["dataset"],
                "k": str(trial["k"]),
                "draw": "" if trial["draw"] is None else str(trial["draw"]),
                "control": str(trial["permuted"]),
                "calibration_view": trial["calibration_view"],
                "query_view": trial["query_view"],
                "true": records[i]["letter"],
            }
            label = trial["trial_id"]
            require(
                all(row[k] == v for k, v in expected.items()), f"Metadata ledger berbeda: {label}"
            )
            require(i not in correct[label], f"Ketukan berulang dalam trial: {label}")
            require(row["replay"] == row["prediction"], f"Prediksi audit berbeda: {label}")
            correct[label][i] = int(row["prediction"] == row["true"])

    for label, trial in by_id.items():
        values, metrics = correct[label], trial["query"]
        require(values.keys() == set(trial["query_indices"]), f"Query tidak sama: {label}")
        require(len(values) == metrics["examples"], f"Jumlah query berbeda: {label}")
        require(sum(values.values()) == metrics["correct"], f"Jumlah benar berbeda: {label}")
        require(
            math.isclose(fmean(values.values()), metrics["accuracy"]), f"Akurasi berbeda: {label}"
        )
    repeats = sum(len(v) for v in correct.values())
    require(repeats == result["predictions_repeated"], "Jumlah baris ledger berbeda.")
    return (
        trials,
        records,
        correct,
        {
            "input_sha256": {
                str(path.relative_to(ROOT)): digest(path)
                for path in (*files, audit_path, Path(__file__).resolve())
            },
            "verified_trials": len(trials),
            "verified_repeated_rows": repeats,
            "hear_ids_with_multiple_device_tokens": read_json(hear_path)[
                "participant_ids_with_multiple_device_tokens"
            ],
        },
    )


def effects(diffs, labels):
    """Summarize paired differences; every draw uses the same observed events."""
    n = len(labels)
    require(all(len(d) == n for d in diffs), "Panjang selisih berpasangan berbeda.")
    draw_means = [fmean(d) for d in diffs]
    event_sums = list(map(sum, zip(*diffs, strict=True)))
    by_class = defaultdict(list)
    for letter, score in zip(labels, event_sums, strict=True):
        by_class[letter].append(score / len(diffs))
    class_means = [fmean(scores) for scores in by_class.values()]
    leave_out = [
        fmean(score for letter, scores in by_class.items() if letter != omitted for score in scores)
        for omitted in by_class
    ]
    return {
        "difference": fmean(draw_means),
        "by_draw": dict(zip(DRAWS, draw_means, strict=True)),
        "draw_range": [min(draw_means), max(draw_means)],
        "macro_difference": fmean(class_means),
        "positive_classes": sum(v > 0 for v in class_means),
        "negative_classes": sum(v < 0 for v in class_means),
        "positive_events": sum(v > 0 for v in event_sums),
        "zero_events": sum(v == 0 for v in event_sums),
        "negative_events": sum(v < 0 for v in event_sums),
        "drop_one_class_range": [min(leave_out), max(leave_out)],
    }


def codec_contrasts(trials, records, correct):
    subset = [t for t in trials if t["id"] == "nc1:fnatic_quiet" and not t["permuted"]]
    cells = {(t["k"], t["draw"], t["calibration_view"], t["query_view"]): t for t in subset}
    require(cells.keys() == set(product(BUDGETS, DRAWS, VIEWS, VIEWS)), "Grid NC1 tidak lengkap.")
    require(len(cells) == len(subset), "Sel NC1 ganda.")
    indices = subset[0]["query_indices"]
    labels = [records[i]["letter"] for i in indices]
    calibration = {t["calibration"] for t in subset}
    query = {t["query_group"] for t in subset}
    require(len(calibration) == len(query) == 1 and calibration != query, "Grup NC1 berbeda.")
    for t in subset:
        require(t["query_indices"] == indices, "Ketukan uji tidak identik.")
        ref = cells[t["k"], t["draw"], ORIGINAL, ORIGINAL]
        require(t["support_indices"] == ref["support_indices"], "Dukungan ketukan tidak identik.")

    def contrast(k, a, b):
        diffs = [
            [
                correct[cells[k, d, *a]["trial_id"]][i] - correct[cells[k, d, *b]["trial_id"]][i]
                for i in indices
            ]
            for d in DRAWS
        ]
        return effects(diffs, labels)

    return {
        "calibration_group": next(iter(calibration)),
        "query_group": next(iter(query)),
        "participants": len({t["subject"] for t in subset}),
        "unique_query_events": len(indices),
        "query_classes_present": len(set(labels)),
        "budgets": [
            {
                "k": k,
                "fit_labels": cells[k, DRAWS[0], ORIGINAL, ORIGINAL]["target_fit_labels"],
                "matched_advantage_original_query": contrast(
                    k, (ORIGINAL, ORIGINAL), (CODEC, ORIGINAL)
                ),
                "matched_advantage_codec_query": contrast(k, (CODEC, CODEC), (ORIGINAL, CODEC)),
                "both_transformed_minus_both_original": contrast(
                    k, (CODEC, CODEC), (ORIGINAL, ORIGINAL)
                ),
            }
            for k in BUDGETS
        ],
    }


def hear_sensitivity(trials):
    subsets = defaultdict(list)
    for t in trials:
        if t["dataset"] == "hear":
            subsets[t["id"]].append(t)
    pairs = []
    expected = {
        (0, None, False),
        *product(BUDGETS, DRAWS, (False,)),
        *product((10,), DRAWS, (True,)),
    }
    for rows in subsets.values():
        by_trial = {(t["k"], t["draw"], t["permuted"]): t for t in rows}
        require(len(rows) == len(by_trial) and by_trial.keys() == expected, "Grid HEAR berbeda.")
        ref = by_trial[0, None, False]
        require(
            all(r["query_indices"] == ref["query_indices"] for r in rows), "Query HEAR bergeser."
        )
        pairs.append(
            {
                **ref,
                "k0": ref["query"]["accuracy"],
                "k10": fmean(by_trial[10, d, False]["query"]["accuracy"] for d in DRAWS),
                "control_k10": fmean(by_trial[10, d, True]["query"]["accuracy"] for d in DRAWS),
            }
        )

    def scope(rows):
        people = defaultdict(list)
        for p in rows:
            people[p["subject"]].append(p)
        per_id = [
            {
                "id": i,
                "pairs": len(ps),
                **{m: fmean(p[m] for p in ps) for m in ("k0", "k10", "control_k10")},
            }
            for i, ps in sorted(people.items())
        ]
        delta = [p["k10"] - p["k0"] for p in per_id]
        loo = (
            [fmean(delta[:i] + delta[i + 1 :]) for i in range(len(per_id))]
            if len(per_id) > 1
            else []
        )
        return {
            "participants": len(people),
            "pairs": len(rows),
            "calibration_groups": len({p["calibration"] for p in rows}),
            "query_groups": len({p["query_group"] for p in rows}),
            "query_events": len({i for p in rows for i in p["query_indices"]}),
            "k10_minus_k0": fmean(delta),
            "drop_one_participant_range": [min(loo), max(loo)] if loo else None,
            "per_id": per_id,
        }

    return {
        "primary": scope([p for p in pairs if p["primary"]]),
        "all": scope(pairs),
        "at_least_12_hours": scope([p for p in pairs if p["gap_hours"] >= 12]),
    }


def pp(value):
    return f"{100 * value:+.2f}".replace(".", ",")


def span(values):
    return f"{pp(values[0])} hingga {pp(values[1])}"


def write_report(summary, output):
    nc1, hear = summary["nc1"], summary["hear"]
    budgets = nc1["budgets"]
    a_key, b_key = "matched_advantage_original_query", "matched_advantage_codec_query"
    a, b = budgets[-1][a_key], budgets[-1][b_key]
    positive_a, positive_b = a["positive_classes"], b["positive_classes"]
    classes, fit_labels = nc1["query_classes_present"], budgets[-1]["fit_labels"]
    draws_a, draws_b = span(a["draw_range"]), span(b["draw_range"])
    omitted_a, omitted_b = span(a["drop_one_class_range"]), span(b["drop_one_class_range"])
    multi_device = summary["hear_ids_with_multiple_device_tokens"]
    verified_rows = f"{summary['verified_repeated_rows']:,}".replace(",", ".")
    lines = [
        "# Audit bukti transfer antar kanal",
        "",
        "Analisis tambahan atas evaluasi HEAR dan NC1 yang sudah selesai. "
        "Memakai ledger evaluasi terverifikasi; tidak ada rekaman, pelatihan, atau prediksi baru.",
        "",
        "## NC1: efek muncul pada kedua arah",
        "",
        f"Semua perbandingan memakai {nc1['unique_query_events']} ketukan uji yang sama dari "
        f"{nc1['participants']} peserta dan satu kelompok tanggal uji `{nc1['query_group']}`. "
        "Dalam tiap K dan pengambilan sampel, ketukan kalibrasi identik di keempat kondisi. "
        "Perbedaan hanya pada ada atau tidaknya pemrosesan codec offline.",
        "",
        "| K | Keuntungan kanal cocok pada uji asli (pp) | Keuntungan pada uji codec (pp) "
        "| Keduanya codec dikurangi keduanya asli (pp) |",
        "|---:|---:|---:|---:|",
        *[
            f"| {r['k']} | {pp(r[a_key]['difference'])} | {pp(r[b_key]['difference'])} "
            f"| {pp(r['both_transformed_minus_both_original']['difference'])} |"
            for r in budgets
        ],
        "",
        f"Pada K10 ({fit_labels} label per pengambilan sampel), rata-rata ketepatan naik pada "
        f"{positive_a} dari {classes} kelas yang hadir untuk uji asli, "
        f"dan {positive_b} untuk uji codec. "
        "Setelah satu kelas mana pun dikeluarkan, "
        f"selisih tetap {omitted_a} pp dan {omitted_b} pp. "
        f"Per pengambilan sampel, selisihnya {draws_a} pp dan {draws_b} pp. "
        "Rentang ini menunjukkan variasi di dalam data yang sama; "
        "kita belum punya interval keyakinan antar sesi.",
        "",
        "![Selisih akurasi per jumlah contoh](selisih-nc1.png)",
        "",
        "## HEAR: pasangan tambahan memakai ulang peserta dan rekaman",
        "",
        "| Cakupan | Pasangan | ID peserta | Grup uji unik | Ketukan uji unik "
        "| K10 dikurangi K0 (pp) | Rentang setelah mengeluarkan 1 ID (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {name} | {r['pairs']} | {r['participants']} | {r['query_groups']} "
            f"| {r['query_events']} | {pp(r['k10_minus_k0'])} "
            f"| {span(r['drop_one_participant_range'])} |"
            for name, r in (
                ("Utama", hear["primary"]),
                ("Semua", hear["all"]),
                ("Jeda ≥12 jam", hear["at_least_12_hours"]),
            )
        ],
        "",
        "Selisih dihitung pada ketukan uji yang sama; rerata pasangan dihitung per ID sebelum "
        "rerata antar ID. K0 memakai model sumber tanpa label HEAR; K10 memakai label dari sesi "
        "HEAR lain. Karena sumber pelatihannya berbeda, perbandingan ini tidak mengisolasi "
        "dampak menambah label pada satu model tetap. Cakupan ≥12 jam memiliki komposisi ID "
        "berbeda, sehingga tidak mengisolasi efek jeda waktu.",
        "",
        "## Batas kesimpulan",
        "",
        "Codec offline pada satu setup memberi keuntungan besar saat kondisi rekam kalibrasi "
        "cocok dengan kondisi uji. Selisih dua kondisi yang sama-sama cocok jauh lebih kecil. "
        "Ini tidak menetapkan efek pergantian mikrofon fisik: NC1 menggunakan satu mikrofon, "
        f"dan audit HEAR menemukan {multi_device} ID dengan beberapa token perangkat. "
        "Jumlah contoh yang diperlukan untuk mikrofon fisik lain masih belum terukur.",
        "",
        "Pemeriksaan: hash input cocok. Kami menghitung ulang jumlah prediksi benar pada "
        f"{summary['verified_trials']} percobaan dari {verified_rows} baris catatan evaluasi. "
        "Satu ketukan bisa muncul di beberapa baris.",
        "",
        "[Ringkasan dan hash input](summary.json).",
    ]
    (output / "laporan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_nc1(nc1, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.3), layout="constrained")
    for key, label, color in (
        ("matched_advantage_original_query", "Kanal cocok, uji asli", "#155b91"),
        ("matched_advantage_codec_query", "Kanal cocok, uji codec", "#84519a"),
        ("both_transformed_minus_both_original", "Keduanya codec − keduanya asli", "#525a62"),
    ):
        rows = [r[key] for r in nc1["budgets"]]
        ax.plot(
            BUDGETS, [100 * r["difference"] for r in rows], "o-", label=label, color=color, lw=2
        )
        ax.fill_between(
            BUDGETS,
            *([100 * r["draw_range"][i] for r in rows] for i in (0, 1)),
            color=color,
            alpha=0.12,
        )
    ax.set(
        title=f"Kecocokan kanal pada {nc1['unique_query_events']} ketukan yang sama",
        xlabel="Contoh kalibrasi per huruf",
        ylabel="Selisih akurasi (poin persentase)",
        xticks=BUDGETS,
    )
    ax.axhline(0, color="#a4aab0", lw=1)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper left", frameon=False, title="Pita: rentang 3 pengambilan sampel")
    fig.savefig(output / "selisih-nc1.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    trials, records, correct, verification = load_ledger()
    summary = {
        **verification,
        "nc1": codec_contrasts(trials, records, correct),
        "hear": hear_sensitivity(trials),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    plot_nc1(summary["nc1"], args.output)
    write_report(summary, args.output)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Audit transfer selesai: {args.output}")


if __name__ == "__main__":
    main()
