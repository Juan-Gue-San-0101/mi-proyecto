#!/usr/bin/env python3
import os, time
import numpy as np
import torch
import onnxruntime as ort
from Vessel_models import UNet2D
from export_onnx import cure_model


def bench_pytorch(model, n=20, tile=256):
    x = torch.randn(1, 3, tile, tile)
    with torch.no_grad():
        for _ in range(3): model(x)
        t0 = time.time()
        for _ in range(n): model(x)
        return (time.time() - t0) / n * 1000


def bench_onnx(path, n=20, tile=256):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 3, tile, tile).astype(np.float32)
    name = sess.get_inputs()[0].name
    for _ in range(3): sess.run(None, {name: x})
    t0 = time.time()
    for _ in range(n): sess.run(None, {name: x})
    return (time.time() - t0) / n * 1000


def main():
    print("Cargando PyTorch...")
    model = UNet2D(3, 3, base_channels=24)
    state = torch.load("weights/model_best_ep20.pth", map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model = cure_model(model, use_int8=False).eval()

    print("=" * 68)
    print("BENCHMARK: PyTorch vs ONNX Runtime")
    print("=" * 68)

    for tile in [256, 512]:
        print(f"\n--- Tile {tile}x{tile} ---")
        t_pt = bench_pytorch(model, n=20, tile=tile)
        print(f"  PyTorch CPU:     {t_pt:8.1f} ms")

        for name, path in [("FP32", "model_best_ep20_fp32_consolidated.onnx"),
                            ("INT8", "model_best_ep20_int8_consolidated.onnx")]:
            if not os.path.exists(path):
                print(f"  ONNX {name}:      (no existe)")
                continue
            t = bench_onnx(path, n=20, tile=tile)
            print(f"  ONNX {name}:      {t:8.1f} ms  ({t_pt/t:.2f}x)")


if __name__ == "__main__":
    main()
