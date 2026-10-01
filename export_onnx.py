#!/usr/bin/env python3
"""
export_onnx.py
--------------
Convierte un checkpoint PyTorch QAT de UNet2D a ONNX "limpio".
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def quantize_weights_permanent(module, frac_bits=7):
    scale = 2.0 ** (-frac_bits)
    qmin, qmax = -128, 127
    with torch.no_grad():
        module.weight.data = (
            (module.weight.data / scale).clamp(qmin, qmax).round() * scale
        )


def cure_model(model, use_int8=False):
    from Vessel_models import QATConv2d, QATConvTranspose2d, FakeQuantAct
    replacements = {}
    for name, module in model.named_modules():
        if isinstance(module, QATConv2d):
            new = nn.Conv2d(
                module.in_channels, module.out_channels,
                module.kernel_size, module.stride, module.padding,
                module.dilation, module.groups, module.bias is not None,
            )
            new.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)
            if use_int8:
                quantize_weights_permanent(new, frac_bits=module.weight_frac_bits)
            replacements[name] = new
        elif isinstance(module, QATConvTranspose2d):
            new = nn.ConvTranspose2d(
                module.in_channels, module.out_channels,
                module.kernel_size, module.stride, module.padding,
                module.output_padding, module.groups, module.bias is not None,
                module.dilation,
            )
            new.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)
            if use_int8:
                quantize_weights_permanent(new, frac_bits=module.weight_frac_bits)
            replacements[name] = new
        elif isinstance(module, FakeQuantAct):
            replacements[name] = nn.Identity()

    for name, new_module in replacements.items():
        parent_name, _, child_name = name.rpartition(".")
        parent = dict(model.named_modules())[parent_name] if parent_name else model
        setattr(parent, child_name, new_module)
    return model


def validate_onnx(model_pt, onnx_path, input_shape=(1, 3, 256, 256),
                  atol=1e-4, rtol=1e-3):
    import onnxruntime as ort
    model_pt.eval()
    dummy = torch.randn(*input_shape)
    with torch.no_grad():
        out_pt = model_pt(dummy).numpy()
    sess = ort.InferenceSession(str(onnx_path),
                                 providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    out_onnx = sess.run(None, {input_name: dummy.numpy()})[0]
    diff = np.abs(out_pt - out_onnx).max()
    close = np.allclose(out_pt, out_onnx, atol=atol, rtol=rtol)
    return diff, close


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--output", default="model.onnx")
    p.add_argument("--base-channels", type=int, default=24)
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--no-stem", action="store_true")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--dynamic", action="store_true", default=True)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    from Vessel_models import UNet2D

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        print(f"ERROR: no existe {weights_path}")
        sys.exit(1)

    print("=" * 68)
    print("EXPORT UNet2D QAT -> ONNX")
    print("=" * 68)
    print(f"Pesos   : {weights_path}")
    print(f"Salida  : {args.output}")
    print(f"Modo    : {'INT8' if args.int8 else 'FP32'}")
    print(f"Opset   : {args.opset}")

    print("\n[1/5] Cargando arquitectura...")
    model = UNet2D(in_channels=3, out_channels=args.num_classes,
                   base_channels=args.base_channels,
                   use_stem=not args.no_stem)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"      Params: {n_par:,} ({n_par * 4 / 1024**2:.2f} MB FP32)")

    print("\n[2/5] Cargando pesos...")
    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    print(f"      OK ({len(state)} tensores)")

    print("\n[3/5] Curando QAT -> estandar...")
    model = cure_model(model, use_int8=args.int8)
    model.eval()
    n_qat = sum(1 for m in model.modules()
                if type(m).__name__ in ("QATConv2d", "QATConvTranspose2d", "FakeQuantAct"))
    print(f"      Modulos QAT restantes: {n_qat}")

    if args.dry_run:
        print("\n[dry-run] No se exporta.")
        return

    print(f"\n[4/5] Exportando a ONNX...")
    dummy = torch.randn(1, 3, args.tile, args.tile)
    dynamic_axes = {
        "input":  {0: "batch", 2: "height", 3: "width"},
        "logits": {0: "batch", 2: "height", 3: "width"},
    } if args.dynamic else None

    torch.onnx.export(
        model, dummy, args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )
    size_mb = Path(args.output).stat().st_size / 1024**2
    print(f"      OK ({size_mb:.2f} MB)")

    print("\n[5/5] Validando ONNX vs PyTorch...")
    try:
        diff, close = validate_onnx(model, args.output,
                                     input_shape=(1, 3, args.tile, args.tile))
        print(f"      Max |diff|: {diff:.2e}")
        print(f"      allclose : {close}")
    except Exception as e:
        print(f"      ERROR: {e}")

    print("\n" + "=" * 68)
    print(f"ONNX en: {Path(args.output).resolve()}")
    print("=" * 68)


if __name__ == "__main__":
    main()
