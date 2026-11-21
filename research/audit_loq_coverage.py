"""Measure how assumed timing offsets change usable quota in f558fd22."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from ketik_audio.common import file_hash, fresh_directory, read_json, write_json
from ketik_audio.web_collection import coverage, read_events, replay_timing
from ketik_audio.web_quota import CLASSES, selection
from research.run_loq_transfer import SOURCE, metadata, timing_and_energy

OFFSETS_MS = (-12.5, 0.0, 12.5)


def summarize(events):
    counts = Counter(events.values())
    return {
        "retained": len(events),
        "per_key": {key: counts[key] for key in CLASSES},
        "minimum_per_key": min(counts[key] for key in CLASSES),
    }


def audit(output):
    meta = metadata(SOURCE)
    audio, rate = sf.read(SOURCE / "audio.wav", dtype="float32")
    events = read_events((SOURCE / "events.jsonl").read_bytes(), meta["event_count"])
    _, clock = replay_timing(
        audio,
        rate,
        meta["first_frame"],
        events,
        read_json(SOURCE / "timing.json"),
        meta["capture_issues"],
    )
    fit, centers, energy, threshold = timing_and_energy(audio, events, clock)
    if fit["status"] != "checked":
        raise ValueError("Pemetaan timing acuan gagal diperiksa.")
    selected = selection(events)["selected_sequences"]
    retained, outcomes = {}, {}
    for shift_ms in OFFSETS_MS:
        affine = {**fit["affine"], "intercept": fit["affine"]["intercept"] + shift_ms * rate / 1000}
        candidates = coverage(
            audio, rate, events, {**fit, "affine": affine}, clock, CLASSES, selected
        )
        kept = {}
        for row in candidates["candidate_events"]:
            at = row["frame_estimate"]
            lo, hi = np.searchsorted(centers, [at - 0.015 * rate, at + 0.035 * rate])
            if lo < hi and energy[lo:hi].max() > threshold:
                kept[row["event_seq"]] = row["letter"]
        retained[shift_ms] = kept
        outcomes[str(shift_ms)] = summarize(kept)

    shared = set.intersection(*(set(kept) for kept in retained.values()))
    intersection = summarize({seq: retained[0.0][seq] for seq in shared})
    fresh_directory(output)
    write_json(
        output / "coverage.json",
        {
            "schema": "ketik-loq-coverage-offset-v1",
            "session_id": meta["session_id"],
            "audio_sha256": meta["files"]["audio.wav"],
            "script_sha256": file_hash(Path(__file__)),
            "selected": len(selected),
            "timing_reference_status": fit["status"],
            "offset_ms": outcomes,
            "same_events_at_all_three_offsets": intersection,
        },
    )
    print(f"Potongan lolos pada ketiga posisi: {intersection['retained']}/{len(selected)}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
