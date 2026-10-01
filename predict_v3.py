#!/usr/bin/env python3
"""
predict_v3.py
-------------
Inferencia optimizada de detección de embarcaciones via ONNX Runtime.

Soporta multiples providers (OpenVINO CPU, CUDA GPU, CPU estandar) con la
misma API. Disenado para ser portable entre laptop (AMD + OpenVINO) y PC
con GPU (NVIDIA + CUDA) sin cambios de codigo.

Uso:
    # Laptop (OpenVINO)
    python predict_v3.py --input muestra_test/ --output-dir out_v3 \\
        --provider openvino

    # PC con GPU (CUDA)
    python predict_v3.py --input muestra_test/ --output-dir out_v3 \\
        --provider cuda

    # Video
    python predict_v3.py --input video.mp4 --output-dir out_v3
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")

CLASS_COLORS = {1: (0, 200, 255), 2: (0, 80, 255)}  # BGR


# --------------------------------------------------------------------------- #
# Provider selection
# --------------------------------------------------------------------------- #
PROVIDERS = {
    "openvino": ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
    "cuda":     ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "cpu":      ["CPUExecutionProvider"],
    "auto":     None,  # decide segun disponibilidad
}


def resolve_providers(name: str):
    if name == "auto":
        avail = ort.get_available_providers()
        for cand in ("CUDAExecutionProvider", "OpenVINOExecutionProvider", "CPUExecutionProvider"):
            if cand in avail:
                return [cand, "CPUExecutionProvider"] if cand != "CPUExecutionProvider" else [cand]
        return ["CPUExecutionProvider"]
    return PROVIDERS[name]


# --------------------------------------------------------------------------- #
# Tiling helpers
# --------------------------------------------------------------------------- #
def tile_starts(total: int, tile: int, stride: int):
    """Posiciones de inicio de tiles, con el ultimo pegado al borde."""
    if total <= tile:
        return [0]
    pos = list(range(0, total - tile + 1, stride))
    if pos[-1] != total - tile:
        pos.append(total - tile)
    return pos


def cosine_window(size: int, floor: float = 0.02) -> np.ndarray:
    """Ventana 2D con caida coseno en los bordes, piso aplicado despues."""
    w = np.hanning(size + 2)[1:-1].astype(np.float32)
    return np.maximum(np.outer(w, w), floor).astype(np.float32)


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
class VesselDetector:
    """
    Wrapper de ONNX Runtime con tiling optimizado y post-proceso.
    """

    def __init__(self, onnx_path: str, provider: str = "auto",
                 tile: int = 256, overlap: int = 32, batch: int = 16,
                 num_classes: int = 3, threshold: float | None = None,
                 min_area: int = 32, expand_px: int = 2):
        self.tile = tile
        self.overlap = overlap
        self.stride = max(1, tile - overlap)
        self.batch = batch
        self.num_classes = num_classes
        self.threshold = threshold
        self.min_area = min_area
        self.expand_px = expand_px

        # Cache cosine window
        self.win = cosine_window(tile)

        # ONNX session
        providers = resolve_providers(provider)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.providers_used = self.sess.get_providers()

    # ----------------------------------------------------------------------- #
    def _forward_batch(self, patches: np.ndarray) -> np.ndarray:
        """Forward ONNX. patches: (B, 3, H, W) float32 en [0,1]."""
        return self.sess.run(None, {self.input_name: patches})[0]

    # ----------------------------------------------------------------------- #
    def predict_probs(self, bgr: np.ndarray) -> np.ndarray:
        """Devuelve probs (num_classes, H, W) para un frame completo."""
        H, W = bgr.shape[:2]
        tile = self.tile
        stride = self.stride

        # Preprocesar: BGR -> RGB -> [0,1] -> CHW
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Padding si el frame es mas chico que el tile
        pad_b = max(0, tile - H)
        pad_r = max(0, tile - W)
        if pad_b or pad_r:
            rgb = cv2.copyMakeBorder(rgb, 0, pad_b, 0, pad_r, cv2.BORDER_REFLECT_101)
        Hp, Wp = rgb.shape[:2]

        chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))

        acc = np.zeros((self.num_classes, Hp, Wp), dtype=np.float32)
        wsum = np.zeros((Hp, Wp), dtype=np.float32)

        # Coordenadas de tiles
        ys = tile_starts(Hp, tile, stride)
        xs = tile_starts(Wp, tile, stride)
        coords = [(y, x) for y in ys for x in xs]
        n_tiles = len(coords)

        # Procesar en batches, pero acumular directo sin list comprehension grande
        for i in range(0, n_tiles, self.batch):
            chunk = coords[i:i + self.batch]
            B = len(chunk)

            # Buffer pre-asignado
            patches = np.empty((B, 3, tile, tile), dtype=np.float32)
            for j, (y, x) in enumerate(chunk):
                patches[j] = chw[:, y:y + tile, x:x + tile]

            probs = self._forward_batch(patches)  # (B, C, H, W)

            # Blending vectorizado por tile
            for j, (y, x) in enumerate(chunk):
                acc[:, y:y + tile, x:x + tile] += probs[j] * self.win
                wsum[y:y + tile, x:x + tile] += self.win

        acc /= np.maximum(wsum, 1e-6)
        return acc[:, :H, :W]

    # ----------------------------------------------------------------------- #
    def probs_to_labels(self, probs: np.ndarray) -> np.ndarray:
        """Argmax o threshold sobre P(foreground)."""
        if self.threshold is None:
            return probs.argmax(0).astype(np.uint8)
        fg = 1.0 - probs[0]
        lab = probs[1:].argmax(0).astype(np.uint8) + 1
        return np.where(fg > self.threshold, lab, 0).astype(np.uint8)

    # ----------------------------------------------------------------------- #
    def labels_to_boxes(self, labels: np.ndarray, shape=None):
        """Componentes conexas -> cajas."""
        H, W = shape if shape else labels.shape
        out = []
        for c in range(1, self.num_classes):
            binary = (labels == c).astype(np.uint8)
            if not binary.any():
                continue
            n, _, stats, cent = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for k in range(1, n):
                area = int(stats[k, cv2.CC_STAT_AREA])
                if area < self.min_area:
                    continue
                x = int(stats[k, cv2.CC_STAT_LEFT]) - self.expand_px
                y = int(stats[k, cv2.CC_STAT_TOP]) - self.expand_px
                w = int(stats[k, cv2.CC_STAT_WIDTH]) + 2 * self.expand_px
                h = int(stats[k, cv2.CC_STAT_HEIGHT]) + 2 * self.expand_px
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(W, x + w), min(H, y + h)
                if x2 <= x1 or y2 <= y1:
                    continue
                out.append({
                    "class_id": c,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "area_px": area,
                    "cx": round(float(cent[k][0]), 2),
                    "cy": round(float(cent[k][1]), 2),
                })
        return out

    # ----------------------------------------------------------------------- #
    def predict(self, bgr: np.ndarray):
        """Pipeline completo: probs -> labels -> boxes."""
        probs = self.predict_probs(bgr)
        labels = self.probs_to_labels(probs)
        boxes = self.labels_to_boxes(labels, bgr.shape[:2])
        return probs, labels, boxes


# --------------------------------------------------------------------------- #
# Visualizacion y escritura
# --------------------------------------------------------------------------- #
def draw_overlay(bgr, labels, boxes, class_names, alpha=0.45):
    vis = bgr.copy()
    tint = np.zeros_like(bgr)
    for c, color in CLASS_COLORS.items():
        tint[labels == c] = color
    mask_any = labels > 0
    vis[mask_any] = cv2.addWeighted(bgr, 1 - alpha, tint, alpha, 0)[mask_any]
    for b in boxes:
        color = CLASS_COLORS.get(b["class_id"], (255, 255, 255))
        cv2.rectangle(vis, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, 2)
        name = class_names[b["class_id"]] if b["class_id"] < len(class_names) else "?"
        cv2.putText(vis, name, (b["x1"], max(12, b["y1"] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return vis


def save_outputs(stem, bgr, labels, boxes, out_dir, class_names, viz=False):
    if viz:
        cv2.imwrite(str(out_dir / f"{stem}_overlay.jpg"),
                    draw_overlay(bgr, labels, boxes, class_names),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    with open(out_dir / f"{stem}_boxes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class_id", "class_name", "x1", "y1", "x2", "y2", "cx", "cy", "area_px"])
        for b in boxes:
            name = class_names[b["class_id"]] if b["class_id"] < len(class_names) else "?"
            w.writerow([b["class_id"], name, b["x1"], b["y1"], b["x2"], b["y2"],
                        b["cx"], b["cy"], b["area_px"]])


def summarize(labels, boxes, class_names, num_classes):
    total = labels.size
    parts = []
    for c in range(1, num_classes):
        px = int((labels == c).sum())
        n = sum(1 for b in boxes if b["class_id"] == c)
        parts.append(f"{class_names[c]}: {n} obj, {px:,} px ({px / total * 100:.3f}%)")
    return "  |  ".join(parts)


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #
def run_images(det, paths, out_dir, class_names, viz=False):
    all_rows = {}
    total_time = 0.0
    for i, path in enumerate(paths, 1):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  [warn] no se pudo leer {path.name}")
            continue
        t0 = time.perf_counter()
        probs, labels, boxes = det.predict(bgr)
        dt = time.perf_counter() - t0
        total_time += dt
        save_outputs(path.stem, bgr, labels, boxes, out_dir, class_names, viz)
        all_rows[path.name] = boxes
        print(f"  [{i}/{len(paths)}] {path.name}  {bgr.shape[1]}x{bgr.shape[0]}  "
              f"{dt:.3f}s  ->  {summarize(labels, boxes, class_names, det.num_classes)}")

    with open(out_dir / "detections.json", "w", encoding="utf-8") as fh:
        json.dump({"class_names": class_names, "detections": all_rows}, fh, indent=2)

    n = len(all_rows)
    if n:
        avg = total_time / n
        print(f"\n  Total: {total_time:.2f}s, Promedio: {avg:.3f}s/img ({1/avg:.1f} FPS)")
    print(f"  Detecciones agregadas: {out_dir / 'detections.json'}")


def run_video(det, path, out_dir, class_names, frame_stride=1):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"ERROR: no se pudo abrir {path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {W}x{H} @ {fps:.1f} fps, {n_frames} frames, stride={frame_stride}")

    writer = cv2.VideoWriter(str(out_dir / f"{path.stem}_annotated.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / frame_stride, (W, H))

    rows = []
    idx = kept = 0
    t_start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_stride != 0:
            idx += 1
            continue
        _, labels, boxes = det.predict(frame)
        writer.write(draw_overlay(frame, labels, boxes, class_names))
        for b in boxes:
            rows.append({"frame": idx, **b})
        kept += 1
        if kept % 10 == 0:
            el = time.perf_counter() - t_start
            print(f"\r  frame {idx}/{n_frames}  {kept / el:.2f} FPS", end="", flush=True)
        idx += 1

    cap.release()
    writer.release()
    el = time.perf_counter() - t_start
    print(f"\n  Procesados {kept} frames en {el:.1f}s ({kept / el:.2f} FPS)")

    with open(out_dir / f"{path.stem}_tracks.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "class_id", "class_name", "x1", "y1", "x2", "y2",
                    "cx", "cy", "area_px"])
        for r in rows:
            name = class_names[r["class_id"]] if r["class_id"] < len(class_names) else "?"
            w.writerow([r["frame"], r["class_id"], name, r["x1"], r["y1"],
                        r["x2"], r["y2"], r["cx"], r["cy"], r["area_px"]])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Inferencia ONNX optimizada de embarcaciones")
    p.add_argument("--model", default="model_best_ep20_int8_consolidated.onnx")
    p.add_argument("--input", required=True, help="Imagen, carpeta o video")
    p.add_argument("--output-dir", default="predictions_v3")
    p.add_argument("--provider", default="auto",
                   choices=["auto", "openvino", "cuda", "cpu"])
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("-b", "--batch-size", type=int, default=16)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--min-area", type=int, default=32)
    p.add_argument("--expand-px", type=int, default=2)
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--class-names", default="fondo,boat,ship")
    p.add_argument("--visualization", action="store_true")
    p.add_argument("--frame-stride", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    class_names = [n.strip() for n in args.class_names.split(",")][: args.num_classes]

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"ERROR: no existe {model_path}")
        sys.exit(1)

    inp = Path(args.input)
    if not inp.exists():
        print(f"ERROR: no existe {inp}")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PREDICT V3 - INFERENCIA ONNX OPTIMIZADA")
    print("=" * 72)
    print(f"Modelo      : {model_path}")
    print(f"Entrada     : {inp}")
    print(f"Salida      : {out_dir}")
    print(f"Provider    : {args.provider}")
    print(f"Tile/Overlap: {args.tile} / {args.overlap}")
    print(f"Batch       : {args.batch_size}")
    print(f"Umbral      : {'argmax' if args.threshold is None else args.threshold}")
    print("=" * 72)

    print("\nCargando modelo...")
    det = VesselDetector(
        onnx_path=str(model_path),
        provider=args.provider,
        tile=args.tile,
        overlap=args.overlap,
        batch=args.batch_size,
        num_classes=args.num_classes,
        threshold=args.threshold,
        min_area=args.min_area,
        expand_px=args.expand_px,
    )
    print(f"Providers activos: {det.providers_used}")
    print()

    if inp.is_dir():
        paths = [p for p in sorted(inp.iterdir()) if p.suffix.lower() in IMG_EXTS]
        if not paths:
            print(f"ERROR: no hay imagenes en {inp}")
            sys.exit(1)
        print(f"Procesando {len(paths)} imagenes...")
        run_images(det, paths, out_dir, class_names, args.visualization)
    elif inp.suffix.lower() in VIDEO_EXTS:
        print("Procesando video...")
        run_video(det, inp, out_dir, class_names, args.frame_stride)
    elif inp.suffix.lower() in IMG_EXTS:
        run_images(det, [inp], out_dir, class_names, args.visualization)
    else:
        print(f"ERROR: extension no reconocida: {inp.suffix}")
        sys.exit(1)

    print("\n" + "=" * 72)
    print("COMPLETADO")
    print("=" * 72)


if __name__ == "__main__":
    main()
