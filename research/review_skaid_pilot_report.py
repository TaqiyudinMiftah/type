"""Publish a readable presentation revision while preserving the signed original report."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ketik_audio.common import ResearchError, file_hash, fresh_directory, now, read_json, write_json
from ketik_audio.skaid_pilot_common import ROOT, verify

SOURCE = ROOT / "reports/skaid-pilot"
OUTPUT = ROOT / "reports/skaid-pilot-reviewed"
LABELS = {
    "mfcc": "MFCC",
    "beats": "BEATs beku",
    "pretrained": "ResNet pralatih",
    "random": "ResNet acak",
}
COLORS = {"mfcc": "#177b58", "beats": "#b97818", "pretrained": "#2d63a5", "random": "#9466a8"}


def readable_text(text):
    result, code = [], False
    for line in text.splitlines():
        if line.startswith("```"):
            code = not code
            result.append(line)
            continue
        if code:
            result.append(line)
            continue
        parts = re.split(r"(`[^`]*`|\[[^\]]*\]\([^)]*\)|\b(?:M4A|L2|K\d+)\b)", line)
        for i in range(0, len(parts), 2):
            fragment = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", parts[i])
            fragment = re.sub(r"(?<=\d)(?=[A-DF-Za-df-z])", " ", fragment)
            fragment = re.sub(r":(?=\S)", ": ", fragment)
            parts[i] = fragment.replace("hingga+", "hingga +")
        result.append("".join(parts))
    return "\n".join(result) + "\n"


def run():
    original = read_json(SOURCE / "summary.json")
    verify(SOURCE, original)
    fresh_directory(OUTPUT)
    signature = {
        str(Path(__file__).relative_to(ROOT)): file_hash(Path(__file__)),
        "reports/skaid-pilot/summary.json": file_hash(SOURCE / "summary.json"),
    }
    write_json(OUTPUT / "lock.json", {"created_at": now(), "input_sha256": signature})
    exclude = {"summary.json", "lock.json", "laporan.md", "kurva-k.png", "kurva-k.pdf"}
    for name in original["artifact_sha256"]:
        if name not in exclude:
            (OUTPUT / name).write_bytes((SOURCE / name).read_bytes())
    groups = {
        (r["family"], r["k"]): r["accuracy"]
        for r in original["analysis"]["groups"]
        if not r["permuted"]
    }
    ks = [1, 2, 3, 5, 10]
    fig, ax = plt.subplots(figsize=(9, 5.8), layout="constrained")
    for family, label in LABELS.items():
        rows = [groups[family, k] for k in ks]
        mean = np.array([r["mean"] for r in rows]) * 100
        ci = np.array([r["ci95_id_bootstrap"] for r in rows]).T * 100
        ax.plot(ks, mean, marker="o", color=COLORS[family], label=label)
        ax.fill_between(ks, ci[0], ci[1], color=COLORS[family], alpha=0.12)
    ax.axhline(100 / 12, color="0.45", linestyle="--", linewidth=1, label="Peluang uniform 1/12")
    ax.set(
        xlabel="K label unik per huruf pada Phase 1",
        ylabel="Akurasi query Phase 2 (%)",
        title="SKAID: kalibrasi 12 huruf pada rekaman terpisah",
        ylim=(0, 100),
    )
    ax.set_xticks(ks)
    upper = ax.secondary_xaxis("top", functions=(lambda x: 12 * x, lambda x: x / 12))
    upper.set_xticks([12 * k for k in ks])
    upper.set_xlabel("Jumlah label dukungan per ID per percobaan")
    ax.legend(loc="upper left", fontsize=9, title="Rerata dan CI95% (17 ID)", title_fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.spines["right"].set_visible(False)
    fig.savefig(OUTPUT / "kurva-k.png", dpi=180)
    fig.savefig(OUTPUT / "kurva-k.pdf")
    plt.close(fig)
    text = readable_text((SOURCE / "laporan.md").read_text())
    text += (
        "\nRevisi penyajian memperbaiki spasi teks dan posisi caption grafik; "
        "semua angka, split, model dan hasil audit berasal dari laporan asli. "
        "Setelah perintah reproduksi di atas, jalankan "
        "` .venv/bin/python research/review_skaid_pilot_report.py ` untuk revisi ini.\n"
    )
    (OUTPUT / "laporan.md").write_text(text, encoding="utf-8")
    verify(SOURCE, original)
    summary = {
        "created_at": now(),
        "status": "complete",
        "input_sha256": signature,
        "presentation_revision_only": True,
        "analysis": original["analysis"],
        "classification_results_unchanged": True,
    }
    summary["artifact_sha256"] = {
        p.name: file_hash(p) for p in sorted(OUTPUT.iterdir()) if p.is_file()
    }
    write_json(OUTPUT / "summary.json", summary)
    return summary


if __name__ == "__main__":
    try:
        run()
    except (ResearchError, OSError, ValueError) as exc:
        raise SystemExit(f"Gagal: {exc}") from exc
