"""Check whether the frozen Harrison evaluation supports claims about sessions."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import fmean, median

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data/external"
RAW = EXTERNAL / "harrison-2023"
PREPARED = EXTERNAL / "harrison-prepared/manifest.json"
FEWSHOT = EXTERNAL / "harrison-fewshot/manifest.json"
RUN = ROOT / "runs/resnet-fewshot"
AUDIT = ROOT / "reports/resnet-fewshot-audit/audit.json"
CONFIGS = [(0.025, 0.02), (0.05, 0.02), (0.10, 0.02), (0.20, 0.02), (0.20, 0.05)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_input(path, expected):
    got = sha(path)
    require(got == expected, f"Input berbeda dari artefak beku: {path.relative_to(ROOT)}")
    return got


def collect_provenance():
    source, prepared = read(FEWSHOT), read(PREPARED)
    verified_input(PREPARED, source["original_manifest_sha256"])
    results, audit = read(RUN / "results.json"), read(AUDIT)
    require(
        results["status"] == "complete" and audit["status"] == "passed",
        "Hasil dan audit harus lengkap.",
    )
    verified_input(RUN / "results.json", audit["results_sha256"])
    verified_input(RUN / "sources-locked.json", audit["source_lock_sha256"])
    require(source["source_commit"] == prepared["commit"], "Commit dataset berbeda.")
    require(results["signature"] == audit["signature"], "Signature evaluasi berbeda.")
    require(source["segments_sha256"] == results["signature"]["segments_sha256"], "Segmen berbeda.")
    verified_input(FEWSHOT.with_name("segments.npz"), source["segments_sha256"])

    records = {(r["domain"], r["key"].casefold()): r for r in prepared["files"]}
    require(
        len(records) == len(prepared["files"]) == len(source["files"]) == 72, "Jumlah WAV berbeda."
    )
    paths = {r["path"] for r in records.values()}
    digests = {r["sha256"] for r in records.values()}
    require(len(paths) == len(digests) == 72, "Path atau digest WAV berulang.")
    for r in source["files"]:
        old = records[r["domain"], r["key"].casefold()]
        require(all(r[k] == v for k, v in old.items()), "Asal atau penanda segmen berbeda.")
    require({d for d, _ in records} == {"phone", "zoom"}, "Rute sumber berbeda.")
    all_ids, markers, tag_names = {}, {}, Counter()
    for (domain, key), record in records.items():
        path = (RAW / record["path"]).resolve(strict=True)
        require(path.is_relative_to(RAW), "Path WAV di luar arsip lokal.")
        verified_input(path, record["sha256"])
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format_tags:stream_tags",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        metadata = json.loads(probe.stdout)
        tags = {
            k.casefold()
            for d in [metadata["format"], *metadata["streams"]]
            for k in d.get("tags", {})
        }
        tag_names.update(tags)
        events = tuple(v / record["sample_rate"] for v in record["centers_samples"])
        require(
            len(events) == 25 and all(a < b for a, b in zip(events[:-1], events[1:], strict=True)),
            "Urutan penanda tidak valid.",
        )
        markers[domain, key] = events
        for i in range(len(events)):
            all_ids[domain, key, i] = record["path"]
    return (
        results,
        audit,
        all_ids,
        markers,
        {
            "source_commit": prepared["commit"],
            "wav_hashes_verified": len(records),
            "ffprobe_tag_names_and_files": dict(tag_names),
        },
    )


def parent_split(results, audit, all_ids):
    def ids(values, domain):
        parts = [v.split(":") for v in values]
        parsed = {(d, key.casefold(), int(i)) for d, key, i in parts}
        require(len(parsed) == len(values), "ID event berulang.")
        require(all(i[0] == domain and i in all_ids for i in parsed), "ID di luar manifest.")
        return parsed

    def counts(support, query):
        require(not support & query, "ID klip yang persis sama dipakai silang split.")
        a, b = ({all_ids[i] for i in block} for block in (support, query))
        return [len(a), len(b), len(a & b)]

    overlap = Counter()
    unique_queries = set()
    k5_scores = []
    for row in results["sources"] + results["conditions"]:
        name = f"{row['directory']}/config.json"
        path = RUN / name
        verified_input(path, audit["artifact_hashes"][name])
        cfg = read(path)
        require(cfg["signature"] == results["signature"], "Konfigurasi di luar run.")
        if row["method"] == "source_only":
            unique_queries.add(frozenset(ids(cfg["test_ids"], "phone")))
            continue
        domain = cfg["domain"]
        query = ids(cfg["test_ids"], domain)
        support = ids(cfg["support_ids"], domain)
        if domain == "zoom":
            support |= ids(cfg["validation_ids"], domain)
        else:
            unique_queries.add(frozenset(query))
        overlap[domain, tuple(counts(support, query))] += 1
        if row["method"] == "adapt" and row["k"] == 5:
            k5_scores.append(row["test"]["accuracy"])
    require(
        len(unique_queries) == 1 and len(next(iter(unique_queries))) == 360, "Query ponsel berubah."
    )
    require(len(k5_scores) == 3, "Blok K5 berbeda.")
    return {
        "configs_verified": len(results["sources"]) + len(results["conditions"]),
        "phone_query_events": len(next(iter(unique_queries))),
        "k5_adaptation_accuracy": fmean(k5_scores),
        "wav_parent_overlap": [
            {
                "domain": domain,
                "fit_parent_count": n_fit,
                "query_parent_count": n_query,
                "common_parent_count": common,
                "fits": count,
            }
            for (domain, (n_fit, n_query, common)), count in sorted(overlap.items())
        ],
    }


def plausible_timeline(phone, zoom, uncertainty, max_rate_change):
    """Return whether ordered times pass a relaxed necessary consistency condition."""
    lo, hi = 1 - max_rate_change, 1 + max_rate_change
    bound = 2 * uncertainty * (1 + hi)
    for (p, z), (q, w) in combinations(zip(phone, zoom, strict=True), 2):
        lo = max(lo, (q - p - bound) / (w - z))
        hi = min(hi, (q - p + bound) / (w - z))
        if lo > hi:
            return False
    return True


def compare_markers(markers):
    keys = {k for d, k in markers if d == "phone"}
    require(keys == {k for d, k in markers if d == "zoom"} and len(keys) == 36, "Kosakata berbeda.")

    def pair(g, k):
        return markers["phone", g], markers["zoom", k]

    drift = []
    for key in keys:
        p, z = pair(key, key)
        drift.append(abs((p[-1] - p[0]) - (z[-1] - z[0])))
    checks = []
    for jitter, speed in CONFIGS:
        same = {k for k in keys if plausible_timeline(*pair(k, k), jitter, speed)}
        other = sum(
            plausible_timeline(*pair(k, g), jitter, speed) for k in keys for g in keys if k != g
        )
        detail = {
            "per_marker_tolerance_ms": int(1000 * jitter),
            "max_relative_rate_change": speed,
            "same_label_files_passing": len(same),
            "same_label_files_passing_without_ambiguous_export": len(same - {"j"}),
            "different_label_files_passing": other,
        }
        if (jitter, speed) in {(0.10, 0.02), (0.20, 0.05)}:
            passing = set()
            for key in keys:
                p, z = pair(key, key)
                if any(
                    plausible_timeline(p[:i] + p[i + 1 :], z[:j] + z[j + 1 :], jitter, speed)
                    for i in range(len(p))
                    for j in range(len(z))
                ):
                    passing.add(key)
            detail["passing_when_omitting_one_per_side"] = len(passing)
            detail["passing_when_omitting_one_per_side_without_ambiguous_export"] = len(
                passing - {"j"}
            )
        checks.append(detail)
    return {
        "marker_pair_screen": checks,
        "same_label_candidates": len(keys),
        "different_label_candidates": len(keys) * (len(keys) - 1),
        "range_and_median_of_first_to_last_elapsed_difference_s": [
            min(drift),
            median(drift),
            max(drift),
        ],
        "verified_same_physical_keystroke_pairs": 0,
    }


def main():
    results, audit, all_ids, markers, provenance = collect_provenance()
    record = {
        **provenance,
        "split": parent_split(results, audit, all_ids),
        "relative_timing": compare_markers(markers),
        "input_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in (PREPARED, FEWSHOT, AUDIT, RUN / "results.json", Path(__file__).resolve())
        },
    }
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
