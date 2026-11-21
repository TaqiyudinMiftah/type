"""Isolated native-runtime probe invoked by devices.gpu_probe."""

import json


def main():
    from .devices import configure_rx6000, is_rx6000

    configure_rx6000()
    result = {
        "available": False,
        "backend": "unknown",
        "name": None,
        "reason": None,
        "stage": "import_torch",
    }
    print(json.dumps(result), flush=True)
    import torch

    backend = "rocm" if torch.version.hip else "cuda" if torch.version.cuda else "cpu"
    result.update(backend=backend, stage="enumerate_gpu")
    print(json.dumps(result), flush=True)
    try:
        if not torch.cuda.is_available():
            result["reason"] = (
                "PyTorch yang terpasang memakai build CPU. Gunakan lingkungan ROCm terpisah "
                "sesuai README untuk menguji GPU AMD."
                if backend == "cpu"
                else "Runtime GPU tidak tersedia atau driver tidak kompatibel."
            )
        else:
            from .models import small_cnn

            torch.set_num_threads(4)
            result["name"] = torch.cuda.get_device_name(0)
            if backend != "rocm" or not is_rx6000(result["name"]):
                result.update(
                    stage="rejected_device",
                    reason=(f"GPU {result['name']} ditolak: proyek hanya memakai Radeon RX 6000."),
                )
                print(json.dumps(result), flush=True)
                return
            result["stage"] = "allocate_model"
            print(json.dumps(result), flush=True)
            model = small_cnn().to("cuda")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            x = torch.ones((2, 1, 64, 48), device="cuda", requires_grad=True)
            result["stage"] = "forward"
            print(json.dumps(result), flush=True)
            loss = model(x).square().mean()
            result["stage"] = "backward"
            print(json.dumps(result), flush=True)
            loss.backward()
            result["stage"] = "optimizer_step"
            print(json.dumps(result), flush=True)
            optimizer.step()
            torch.cuda.synchronize()
            if not torch.isfinite(loss).item() or not torch.isfinite(x.grad).all().item():
                raise RuntimeError("Hasil operasi GPU bukan nilai finite.")
            result["available"] = True
            result["stage"] = "complete"
    except Exception as exc:
        result["reason"] = f"Tes forward/backward GPU gagal: {exc}"
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
