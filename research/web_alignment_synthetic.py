"""Independent generated waveforms with known clock-to-audio geometry."""

from __future__ import annotations

import numpy as np


def make_case(case="clean", seed=7100, rate=16000):
    rng = np.random.default_rng(seed)
    first = 4096
    audio = rng.normal(0, 2e-5, 36 * rate).astype(np.float32)
    offset_s = float(rng.uniform(0.032, 0.075))
    slope = 1 + float(rng.uniform(-250, 250)) * 1e-6 if case == "drift" else 1.0
    steps = [(3.1 + i * 0.83, " ", "sync_start") for i in range(8)]
    steps += [(12 + 2 * i, c, "typing") for i, c in enumerate("abcde")]
    steps += [(27 + i * 0.83, " ", "sync_end") for i in range(8)]
    weak = set(map(int, rng.choice(16, 5, replace=False))) if case == "weak_markers" else set()
    if case == "insufficient_training_support":
        weak = {0, 2, 8, 10}
    events, truth, marker_seq = [], [], []
    ordinal = 0

    def pulse(at, amplitude):
        center = round(at)
        half = max(2, round(rate * 0.0003))
        audio[center - half : center + half + 1] += amplitude

    for seconds, key, phase in steps:
        clock = seconds * rate
        bend = 0.0
        if case == "hidden_middle_bend" and 9 < seconds < 26:
            bend = 0.080 * rate * np.sin(np.pi * (seconds - 9) / 17) ** 2
        physical = slope * clock + offset_s * rate + bend
        seq = len(events)
        row = {
            "seq": seq,
            "type": "keydown",
            "phase": phase,
            "time_ms": 1000 + seconds * 1000,
            "handler_ms": 1000 + seconds * 1000 + 0.1,
            "is_trusted": False,
            "key": key,
            "code": "Space" if key == " " else "Key" + key.upper(),
            "repeat": False,
            "is_composing": False,
            "shift": False,
            "ctrl": False,
            "alt": False,
            "meta": False,
        }
        is_marker = key == " "
        if is_marker:
            row["calibration"] = True
            marker_seq.append(seq)
        if not (is_marker and ordinal in weak) and case not in {"silence", "stationary_noise"}:
            amplitude = float(rng.uniform(0.58, 0.64))
            if is_marker and case == "training_distractor" and ordinal == 0:
                amplitude = 0.12
            pulse(physical, amplitude)
        if (
            is_marker
            and case in {"training_distractor", "training_competing_distractor"}
            and ordinal == 0
        ):
            pulse(clock - 0.100 * rate, 0.98)
        if is_marker and case == "validation_distractor" and ordinal == 1:
            pulse(clock - 0.100 * rate, 0.98)
        if is_marker and case == "competing_peaks":
            pulse(physical + 0.120 * rate, 0.61)
        events.append(row)
        events.append(
            {
                **row,
                "seq": seq + 1,
                "type": "keyup",
                "time_ms": row["time_ms"] + 85,
                "handler_ms": row["handler_ms"] + 85,
            }
        )
        events[-1].pop("calibration", None)
        truth.append(
            {"event_seq": seq, "phase": phase, "clock_frame": clock, "physical_frame": physical}
        )
        if is_marker:
            ordinal += 1
    if case == "silence":
        audio.fill(0)
    elif case == "stationary_noise":
        audio = rng.normal(0, 0.003, len(audio)).astype(np.float32)
    elif case == "missing_end":
        events = events[:-2]
    elif case == "dispatch_lag":
        events[0]["handler_ms"] += 35
    anchors = [
        {"before_ms": 1000 + 500 * i, "after_ms": 1000 + 500 * i, "context_s": first / rate + i / 2}
        for i in range(73)
    ]
    return {
        "audio": audio,
        "rate": rate,
        "first": first,
        "events": events,
        "anchors": anchors,
        "truth": truth,
        "true_slope": slope,
        "true_offset_frames": offset_s * rate,
        "capture_issues": ["window_blurred"] if case == "capture_interrupted" else [],
        "case": case,
        "seed": seed,
        "marker_seq": marker_seq,
    }
