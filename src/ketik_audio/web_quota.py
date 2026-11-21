"""Versioned browser quotas; the terminal dataset keeps its original a-z labels."""

from __future__ import annotations

import math
from collections import Counter

from .common import ResearchError

SCHEMA = "ketik-web-session/v2"
CLASSES = (*"abcdefghijklmnopqrstuvwxyz", "space")
PER_CLASS = 25
MAX_AUDIO_BYTES = 128 * 1024**2
EXTRA_KEYS = (
    "Backspace", "Enter", "Tab", "Delete", "ShiftLeft", "ShiftRight",
    "ControlLeft", "ControlRight", "AltLeft", "AltRight",
    "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown",
)


def config():
    return {
        "protocol": "balanced-27-v1",
        "classes": list(CLASSES),
        "per_class": PER_CLASS,
        "selection": "first-eligible-keydown",
    }


def event_class(event):
    if (
        event["type"] != "keydown"
        or event["phase"] != "typing"
        or any(event[k] for k in ("repeat", "is_composing", "ctrl", "alt", "meta"))
    ):
        return None
    if event["key"] == " " and event["code"] == "Space":
        return "space"
    key = event["key"].lower()
    return key if len(key) == 1 and key in CLASSES else None


def selection(events):
    raw, chosen, sequences = Counter(), Counter(), []
    for event in events:
        key = event_class(event)
        eligible = bool(key and chosen[key] < PER_CLASS)
        if (
            event["type"] == "keydown"
            and (
                type(event.get("quota_selected")) is not bool or event["quota_selected"] != eligible
            )
        ) or (event["type"] != "keydown" and event.get("quota_selected") is True):
            raise ResearchError("Pemilihan kuota berbeda dari tombol aktual.")
        if event["phase"] == "typing" and event.get("collection_mode") not in {"sentence", "topup"}:
            raise ResearchError("Jenis latihan event v2 tidak valid.")
        if key:
            raw[key] += 1
        if eligible:
            chosen[key] += 1
            sequences.append(event["seq"])
    return {
        **config(),
        "raw_counts": {k: raw[k] for k in CLASSES},
        "selected_counts": {k: chosen[k] for k in CLASSES},
        "selected_sequences": sequences,
        "selected_total": len(sequences),
        "complete": all(chosen[k] == PER_CLASS for k in CLASSES),
    }


def validate_preflight(meta):
    def number(v):
        return type(v) in (int, float) and math.isfinite(v)

    def require(condition, text):
        if not condition:
            raise ResearchError(text)

    check = meta.get("preflight")
    require(
        isinstance(check, dict)
        and check.get("method") in {"key-signal-v1", "key-signal-v2", "key-signal-v3"}
        and check.get("status") == "passed"
        and check.get("passed") == check.get("total") == 8,
        "Sesi v2 wajib memuat pemeriksaan suara 8/8 yang lulus.",
    )
    rate = check.get("sample_rate")
    noise = check.get("quiet_rms")
    require(
        type(rate) is int and 8000 <= rate <= 96000 and number(noise) and noise >= 0,
        "Parameter pemeriksaan suara tidak valid.",
    )
    adaptive = check["method"] in {"key-signal-v2", "key-signal-v3"}
    context_mapping = check["method"] == "key-signal-v3"
    if context_mapping:
        require(
            type(check.get("first_frame")) is int and check["first_frame"] >= 0
            and type(check.get("frames")) is int and 0 < check["frames"] <= MAX_AUDIO_BYTES // 4,
            "Rentang audio pemeriksaan suara tidak valid.",
        )
    if "source" in check:
        require(
            check["source"] == "session-sync-start"
            and context_mapping
            and rate == meta.get("sample_rate")
            and check["first_frame"] == meta.get("first_frame")
            and check["frames"] <= meta["frames"],
            "Audio pemeriksaan suara tidak cocok dengan awal sesi.",
        )
    if adaptive:
        require(
            check.get("baseline_estimator") == "median-5ms-ac-rms"
            and number(check.get("min_snr_db")) and check["min_snr_db"] == 6
            and number(check.get("floor_rms")) and check["floor_rms"] == 1e-6
            and number(check.get("quiet_p95_rms")) and check["quiet_p95_rms"] >= noise
            and number(check.get("quiet_raw_rms")) and check["quiet_raw_rms"] >= noise,
            "Parameter pemeriksaan suara adaptif tidak valid.",
        )
        threshold = max(1e-6, noise * 10 ** (6 / 20), check["quiet_p95_rms"] * 1.2)
    else:
        threshold = max(1e-4, 4 * noise)
    markers = check.get("markers")
    require(
        isinstance(markers, list) and len(markers) == 8, "Penanda pemeriksaan suara tidak lengkap."
    )
    require(
        isinstance(meta.get("track_settings"), dict)
        and isinstance(check.get("microphone_label"), str)
        and check["microphone_label"] == meta["track_settings"].get("microphone_label"),
        "Mikrofon sesi berbeda dari pemeriksaan suara.",
    )
    seen = set()
    for marker in markers:
        require(isinstance(marker, dict), "Penanda pemeriksaan suara tidak valid.")
        seq = marker.get("event_seq")
        require(
            type(seq) is int and seq >= 0 and seq not in seen,
            "Urutan penanda pemeriksaan suara tidak valid.",
        )
        seen.add(seq)
        require(
            marker.get("reason") is None
            and all(
                number(marker.get(k))
                for k in ("peak_rms", "peak_abs", "threshold", "clock_frame", "peak_frame")
            ),
            "Nilai pemeriksaan suara tidak valid.",
        )
        require(
            math.isclose(marker["threshold"], threshold, abs_tol=1e-12)
            and marker["peak_rms"] > marker["threshold"]
            and marker["peak_rms"] <= marker["peak_abs"] < 0.999
            and marker["clock_frame"] >= 0
            and marker["peak_frame"] >= 0
            and abs(marker["peak_frame"] - marker["clock_frame"]) <= 0.25 * rate,
            "Pemeriksaan suara mengklaim lulus dengan sinyal yang tidak memadai.",
        )
        if adaptive:
            require(
                type(marker.get("timing_ambiguous")) is bool
                and number(marker.get("snr_db"))
                and math.isclose(
                    marker["snr_db"],
                    20 * math.log10(max(marker["peak_rms"], 1e-12) / max(noise, 1e-9)),
                    abs_tol=1e-8,
                ),
                "Nilai pemeriksaan suara adaptif berbeda dari rasio sinyal terhadap latar.",
            )
        if context_mapping:
            source = marker.get("frame_source")
            require(
                source in {"audio-context-snapshot", "performance-clock-interpolation"}
                and marker["clock_frame"] < check["frames"]
                and marker["peak_frame"] < check["frames"],
                "Pemetaan waktu pemeriksaan suara tidak valid.",
            )
            if source == "audio-context-snapshot":
                require(
                    number(marker.get("event_audio_context_s"))
                    and marker["event_audio_context_s"] >= 0
                    and math.isclose(
                        marker["clock_frame"],
                        marker["event_audio_context_s"] * rate - check["first_frame"],
                        abs_tol=1e-6,
                    ),
                    "Posisi AudioContext pemeriksaan suara berbeda dari frame yang dinilai.",
                )


def validate_stored_check(check, events):
    """Link an embedded sound check to the initial synchronization taps."""
    if check.get("source") != "session-sync-start":
        return
    initial = [e for e in events if e.get("calibration") and e["phase"] == "sync_start"]
    if [e["seq"] for e in initial] != [m["event_seq"] for m in check["markers"]]:
        raise ResearchError("Ketukan pemeriksaan suara berbeda dari penanda awal sesi.")
    for event, marker in zip(initial, check["markers"], strict=True):
        if (
            marker["frame_source"] == "audio-context-snapshot"
            and event.get("audio_context_s") != marker["event_audio_context_s"]
        ):
            raise ResearchError("Waktu pemeriksaan suara berbeda dari event awal sesi.")


def additional_counts(events):
    counts = dict.fromkeys(EXTRA_KEYS, 0)
    for event in events:
        if (
            event["type"] == "keydown" and event["phase"] == "typing"
            and not event["repeat"] and not event["is_composing"]
            and event["code"] in counts
        ):
            counts[event["code"]] += 1
    return {"method": "physical-code-keydown-v1", "scope": "typing", "counts": counts}
