"""Synthetic SOFTWARE verification, never evidence of real keyboard accuracy."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from .common import LABELS, SCHEMA_VERSION, fresh_directory, role_for_session, write_json
from .recording import make_prompts


def synthetic_sessions(root: Path, per_key: int = 2) -> None:
    rate = 16_000
    for number in range(1, 7):
        rng = np.random.default_rng(100 + number)
        prompts = make_prompts(per_key, 100 + number)
        times = np.arange(len(prompts)) * 0.85 + 3.0
        audio = rng.normal(0, 0.0001, round((times[-1] + 1) * rate)).astype(np.float32)
        rows = []
        for key, seconds in zip(prompts, times, strict=True):
            t = np.arange(round(rate * 0.12)) / rate
            frequency = 200 + LABELS.index(key) * 135 + rng.normal(0, 1)
            envelope = np.exp(-0.5 * ((t - 0.04) / 0.012) ** 2)
            wave = (
                0.2
                * envelope
                * (np.sin(2 * np.pi * frequency * t) + 0.3 * np.sin(2 * np.pi * 2 * frequency * t))
            )
            start = round(seconds * rate)
            audio[start : start + len(wave)] += wave.astype(np.float32)
            rows.append(
                {
                    "key": key,
                    "expected": key,
                    "sample_index": start + round(0.025 * rate),
                    "accepted": 1,
                    "reason": "synthetic_software_test",
                }
            )
        path = root / "raw" / f"session-{number:02d}"
        fresh_directory(path)
        sf.write(path / "audio.wav", audio, rate, subtype="FLOAT")
        with (path / "events.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(
            path / "session.json",
            {
                "schema_version": SCHEMA_VERSION,
                "session": number,
                "role": role_for_session(number),
                "status": "complete",
                "synthetic": True,
                "per_key": per_key,
                "sample_rate": rate,
                "frames": len(audio),
                "accepted_events": len(rows),
                "input_overflows": 0,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Uji sintetis perangkat lunak, bukan akurasi riset."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    from .dataset import prepare
    from .evaluation import evaluate, predict
    from .models import train_cnn, train_svm
    from .reporting import report

    fresh_directory(args.output)
    root = args.output / "data"
    print("DATA SINTETIS — HANYA VERIFIKASI PERANGKAT LUNAK", flush=True)
    synthetic_sessions(root)
    prepared = root / "processed"
    prepare(root, prepared)
    svm, cnn = args.output / "svm", args.output / "cnn"
    train_svm(prepared, svm)
    train_cnn(prepared, cnn, requested="cpu", epochs=args.epochs)
    evaluate(svm, root)
    evaluate(cnn, root)
    write_json(args.output / "prediction.json", predict(svm, root / "raw/session-06/audio.wav"))
    print(report([svm, cnn], args.output / "report"))


if __name__ == "__main__":
    main()
