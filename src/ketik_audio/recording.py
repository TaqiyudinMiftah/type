"""Linux terminal-only keyboard labels and microphone sample-clock synchronization."""

from __future__ import annotations

import csv
import os
import queue
import random
import select
import sys
import threading
import time
from contextlib import AbstractContextManager
from pathlib import Path

import numpy as np
import soundfile as sf

from .common import (
    LABELS,
    SCHEMA_VERSION,
    ResearchError,
    fresh_directory,
    now,
    read_json,
    role_for_session,
    write_json,
)


def make_prompts(per_key: int, seed: int) -> list[str]:
    if per_key < 1:
        raise ResearchError("--per-key minimal 1.")
    rng = random.Random(seed)
    remaining = {key: per_key for key in LABELS}
    prompts = []
    while any(remaining.values()):
        candidates = [
            key for key in LABELS if remaining[key] and (not prompts or key != prompts[-1])
        ]
        if not candidates:
            candidates = [key for key in LABELS if remaining[key]]
        key = rng.choices(candidates, weights=[remaining[k] for k in candidates])[0]
        prompts.append(key)
        remaining[key] -= 1
    return prompts


class TerminalKeys(AbstractContextManager):
    """No global hooks: only the foreground terminal's own standard input."""

    def __init__(self, fd: int | None = None):
        self.fd = sys.stdin.fileno() if fd is None else fd
        self.saved = None

    def __enter__(self):
        import termios
        import tty

        if not sys.platform.startswith("linux"):
            raise ResearchError("Perekam terminal v1 membutuhkan Linux.")
        if not os.isatty(self.fd):
            raise ResearchError("Jalankan record di terminal interaktif (bukan pipe/notebook).")
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def read(self, timeout: float = 0.1) -> str | None:
        import os

        if select.select([self.fd], [], [], timeout)[0]:
            value = os.read(self.fd, 1)
            if not value:
                raise KeyboardInterrupt
            return value.decode("ascii", errors="replace")
        return None

    def __exit__(self, *args):
        import termios

        if self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
        return False


class AudioCapture(AbstractContextManager):
    def __init__(self, path: Path, device: int | None):
        self.path, self.device = path, device
        self.first_adc: float | None = None
        self.frames = 0
        self.overflows = 0
        self.clipped_samples = 0
        self.queue = queue.SimpleQueue()
        self.error: BaseException | None = None
        self.stream = None
        self.worker = None

    def callback(self, indata, frames, timestamps, status):
        if self.first_adc is None:
            self.first_adc = float(timestamps.inputBufferAdcTime)
        if status.input_overflow:
            self.overflows += 1
        # File I/O stays in the worker; the audio callback only copies a block.
        self.queue.put(indata.copy())

    def _write(self):
        try:
            with sf.SoundFile(
                self.path, mode="w", samplerate=self.rate, channels=1, subtype="FLOAT"
            ) as output:
                while True:
                    block = self.queue.get()
                    if block is None:
                        break
                    self.clipped_samples += int((np.abs(block) >= 0.999).sum())
                    self.frames += len(block)
                    output.write(block)
        except BaseException as exc:
            self.error = exc

    def __enter__(self):
        try:
            import sounddevice as sd
        except (ImportError, OSError) as exc:
            raise ResearchError(f"PortAudio belum tersedia: {exc}. Jalankan ketik doctor.") from exc
        try:
            details = sd.query_devices(self.device, "input")
            self.rate = int(details["default_samplerate"])
            self.device_name = details["name"]
            sd.check_input_settings(
                device=self.device, channels=1, samplerate=self.rate, dtype="float32"
            )
            self.stream = sd.InputStream(
                device=self.device,
                samplerate=self.rate,
                channels=1,
                dtype="float32",
                callback=self.callback,
            )
            self.worker = threading.Thread(target=self._write, daemon=True)
            self.worker.start()
            self.stream.start()
            return self
        except (OSError, RuntimeError, sd.PortAudioError) as exc:
            self._close()
            raise ResearchError(f"Mikrofon gagal dibuka: {exc}. Jalankan ketik doctor.") from exc

    def _close(self):
        try:
            if self.stream is not None:
                self.stream.close()
        finally:
            if self.worker is not None:
                self.queue.put(None)
                self.worker.join(timeout=15)
                if self.worker.is_alive():
                    self.error = RuntimeError("Penulis audio tidak berhenti dalam 15 detik.")

    def __exit__(self, *args):
        self._close()
        if self.error is not None and args[0] is None:
            raise ResearchError(f"Gagal menyimpan audio: {self.error}")
        return False


def sample_index(event_stream_time: float, first_adc_time: float, rate: int) -> int:
    return round((event_stream_time - first_adc_time) * rate)


def record(
    root: Path,
    number: int,
    per_key: int = 25,
    device: int | None = None,
    seed: int = 42,
    retry: bool = False,
) -> dict:
    role = role_for_session(number)
    prompts = make_prompts(per_key, seed + number)
    # Verify audio and terminal prerequisites before creating a raw session.
    if not sys.platform.startswith("linux") or not sys.stdin.isatty():
        raise ResearchError("record membutuhkan terminal interaktif pada laptop Linux.")
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise ResearchError(f"PortAudio belum tersedia: {exc}. Jalankan ketik doctor.") from exc
    try:
        sd.query_devices(device, "input")
    except (OSError, RuntimeError, sd.PortAudioError) as exc:
        raise ResearchError(f"Input mikrofon belum siap: {exc}. Jalankan ketik doctor.") from exc
    path = root / "raw" / f"session-{number:02d}"
    if retry and path.exists():
        old = read_json(path / "session.json")
        if old.get("status") == "complete":
            raise ResearchError("Sesi lengkap tidak dapat ditimpa. Gunakan direktori data baru.")
        archive = root / "failed" / f"session-{number:02d}-{time.time_ns()}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        path.rename(archive)
    fresh_directory(path)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "session": number,
        "role": role,
        "status": "recording",
        "synthetic": False,
        "started_at": now(),
        "per_key": per_key,
        "prompt_seed": seed + number,
        "minimum_interval_ms": 800,
    }
    write_json(path / "session.json", metadata)
    events = []
    ignored_intervals = []
    completed = 0
    capture = AudioCapture(path / "audio.wav", device)
    status = "aborted"
    try:
        with TerminalKeys() as keys, capture:
            print(
                "Diam selama 3 detik untuk merekam kebisingan ruangan. Ctrl+C untuk berhenti.",
                flush=True,
            )
            time.sleep(3)
            # Discard any keys typed during the quiet prefix, without using them as labels.
            while keys.read(0) is not None:
                pass
            last_key_time = float("-inf")
            displayed = None
            while completed < len(prompts):
                if capture.error:
                    raise ResearchError(f"Penulis audio gagal: {capture.error}")
                if capture.overflows:
                    raise ResearchError("Buffer audio hilang. Sesi ditandai invalid; rekam ulang.")
                prompt = prompts[completed]
                if displayed != completed:
                    print(
                        f"\r[{completed + 1}/{len(prompts)}] Tekan {prompt} "
                        "(jeda ≥0,8 dtk; spasi=istirahat): ",
                        end="",
                        flush=True,
                    )
                    displayed = completed
                key = keys.read()
                if key is None:
                    continue
                timestamp = float(capture.stream.time)
                if key == "\x03":
                    raise KeyboardInterrupt
                if key == " ":
                    ignored_intervals.append([timestamp - 0.1, None])
                    print("\nIstirahat. Tekan Enter untuk melanjutkan.", flush=True)
                    while True:
                        control = keys.read()
                        if control == "\x03":
                            raise KeyboardInterrupt
                        if control in ["\r", "\n"]:
                            break
                        if capture.error:
                            raise ResearchError(str(capture.error))
                    last_key_time = float(capture.stream.time)
                    ignored_intervals[-1][1] = last_key_time + 0.35
                    displayed = None
                    continue
                if key not in LABELS:
                    continue
                interval_ok = timestamp - last_key_time >= 0.8
                accepted = key == prompt and interval_ok
                events.append(
                    {
                        "key": key,
                        "expected": prompt,
                        "stream_time": timestamp,
                        "accepted": int(accepted),
                        "reason": "ok"
                        if accepted
                        else "too_fast"
                        if not interval_ok
                        else "wrong_key",
                    }
                )
                last_key_time = timestamp
                if accepted:
                    completed += 1
                else:
                    print("\nUlangi setelah jeda 0,8 detik.", flush=True)
                    displayed = None
            time.sleep(0.4)  # Keep the last key's release in the waveform.
            status = "complete"
    except KeyboardInterrupt:
        print("\nPerekaman dihentikan; sesi disimpan sebagai aborted.", flush=True)
    finally:
        print(flush=True)
        metadata.update(
            {
                "finished_at": now(),
                "status": status,
                "accepted_events": completed,
                "input_overflows": capture.overflows,
                "frames": capture.frames,
                "sample_rate": getattr(capture, "rate", 0),
                "device_name": getattr(capture, "device_name", None),
                "clipped_samples": capture.clipped_samples,
                "clipped_fraction": capture.clipped_samples / max(1, capture.frames),
            }
        )
        if (
            capture.overflows
            or capture.error
            or (status == "complete" and (capture.first_adc is None or not capture.frames))
        ):
            metadata["status"] = "invalid"
        origin = capture.first_adc if capture.first_adc is not None else 0.0
        metadata["ignored_intervals_samples"] = [
            [
                max(0, sample_index(start, origin, getattr(capture, "rate", 0))),
                min(
                    capture.frames,
                    sample_index(end, origin, getattr(capture, "rate", 0)),
                )
                if end is not None
                else capture.frames,
            ]
            for start, end in ignored_intervals
        ]
        with (path / "events.csv").open("w", newline="", encoding="utf-8") as output:
            fields = ["key", "expected", "sample_index", "accepted", "reason"]
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for event in events:
                timestamp = event.pop("stream_time")
                event["sample_index"] = sample_index(timestamp, origin, getattr(capture, "rate", 0))
                writer.writerow(event)
        write_json(path / "session.json", metadata)
    return metadata
