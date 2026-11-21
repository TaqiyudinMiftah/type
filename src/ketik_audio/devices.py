"""Hardware probes and explicit, visible CPU/GPU selection."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

from .common import ResearchError, environment


def configure_rx6000() -> None:
    """Expose only the selected discrete RX 6000; never fall through to the iGPU."""
    ordinal = os.environ.get("KETIK_ROCM_DEVICE", "0")
    if not ordinal.isdecimal():
        raise ResearchError("KETIK_ROCM_DEVICE harus berupa indeks ROCm non-negatif.")
    os.environ["ROCR_VISIBLE_DEVICES"] = ordinal
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
    os.environ["ROCM_SDK_TARGET_FAMILY"] = "gfx1032"


def is_rx6000(name: str) -> bool:
    return bool(re.search(r"\bradeon\s+rx\s*6\d{3}\b", name, re.IGNORECASE))


def gpu_probe() -> dict:
    configure_rx6000()
    result = {"available": False, "backend": "unknown", "name": None, "reason": None}
    try:
        # Unsupported native kernels can abort or hang, so probe in a bounded subprocess.
        process = subprocess.run(
            [sys.executable, "-m", "ketik_audio.gpu_check"],
            capture_output=True,
            text=True,
            timeout=45,
        )
        if process.returncode != 0:
            result["reason"] = (
                f"Proses tes GPU gagal (exit {process.returncode}): "
                + process.stderr.strip()[-1500:]
            )
            return result
        result = json.loads(process.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or b""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        try:
            result = json.loads(output.strip().splitlines()[-1])
        except (ValueError, IndexError):
            pass
        result["reason"] = f"Tes GPU melewati 45 detik pada tahap {result.get('stage', 'unknown')}."
        result["available"] = False
    except (OSError, ValueError, IndexError) as exc:
        result["reason"] = f"Tes forward/backward GPU gagal: {exc}"
    return result


def select_device(request: str) -> tuple[str, dict]:
    if request == "cpu":
        return "cpu", {"requested": "cpu", "selected": "cpu", "fallback_reason": None}
    probe = gpu_probe()
    if probe["available"]:
        return "cuda", {"requested": request, "selected": "gpu", **probe}
    if request == "gpu":
        raise ResearchError(probe["reason"])
    return "cpu", {
        "requested": request,
        "selected": "cpu",
        "fallback_reason": probe["reason"],
        "gpu_probe": probe,
    }


def doctor() -> dict:
    configure_rx6000()
    import torch

    torch.set_num_threads(4)
    audio = {"ready": False, "inputs": [], "reason": None}
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        audio["reason"] = (
            f"PortAudio/mikrofon tidak tersedia: {exc}. Pada laptop Ubuntu, "
            "pasang libportaudio2 lalu jalankan doctor kembali."
        )
    else:
        try:
            for i, device in enumerate(sd.query_devices()):
                if device["max_input_channels"] > 0:
                    audio["inputs"].append(
                        {
                            "id": i,
                            "name": device["name"],
                            "sample_rate": device["default_samplerate"],
                        }
                    )
            audio["ready"] = bool(audio["inputs"])
            if not audio["ready"]:
                audio["reason"] = (
                    "Tidak ada input mikrofon. Jalankan record pada laptop Linux Anda."
                )
        except sd.PortAudioError as exc:
            audio["reason"] = f"Input audio gagal diperiksa: {exc}"
    cpu = {"ready": False, "reason": None}
    try:
        layer = torch.nn.Conv2d(1, 4, 3)
        loss = layer(torch.ones((2, 1, 64, 48))).square().mean()
        loss.backward()
        cpu["ready"] = bool(torch.isfinite(loss))
    except RuntimeError as exc:
        cpu["reason"] = str(exc)
    rocm = {"detected": False, "agents": []}
    if shutil.which("rocminfo"):
        try:
            result = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10)
            rocm["detected"] = result.returncode == 0
            rocm["agents"] = [
                line.strip()
                for line in result.stdout.splitlines()
                if "Marketing Name:" in line or "Device Type:" in line
            ]
        except (OSError, subprocess.TimeoutExpired) as exc:
            rocm["reason"] = str(exc)
    return {
        "environment": environment(),
        "audio": audio,
        "cpu": cpu,
        "gpu": gpu_probe(),
        "rocm": rocm,
        "recording_ready": audio["ready"],
        "training_ready": cpu["ready"],
    }
