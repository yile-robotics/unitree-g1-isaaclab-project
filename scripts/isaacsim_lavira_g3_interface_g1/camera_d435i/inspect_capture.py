#!/usr/bin/env python3
"""在笔记本检查探针保存的RGB-D样本；保留原始数据，另存图表和报告。"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def inspect(sample: Path) -> dict:
    meta = json.loads((sample / "metadata.json").read_text())
    rgb = np.asarray(Image.open(sample / "rgb.ppm").convert("RGB"))
    # Pillow正确解码16位大端PGM；不能用uint8解码或给深度做RGB色彩转换。
    raw = np.asarray(Image.open(sample / "depth_aligned_z16.pgm"))
    if raw.ndim != 2 or raw.shape != rgb.shape[:2] or meta["depth_aligned_to"] != "color":
        raise ValueError("RGB/depth shape or alignment metadata mismatch")
    color = meta["color_intrinsics"]
    aligned = meta["aligned_depth_intrinsics"]
    for intr in (color, aligned):
        if (intr["height"], intr["width"]) != raw.shape:
            raise ValueError("Intrinsics dimensions mismatch")
    if not np.allclose(color["K"], aligned["K"], atol=1e-6, rtol=0):
        raise ValueError("Aligned depth intrinsics do not match color")
    scale = float(meta["depth_scale_m"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid depth scale")
    depth_m = raw.astype(np.float32) * scale
    valid = raw > 0
    if not np.any(valid):
        raise ValueError("No nonzero depth values")
    percentiles = np.percentile(depth_m[valid], [1, 5, 50, 95, 99])
    same_domain = meta["color_timestamp_domain"] == meta["depth_timestamp_domain"]
    report = {
        "serial": meta["serial"], "sdk_version": meta["sdk_version"],
        "firmware": meta["firmware"], "shape_hw": list(raw.shape),
        "depth_scale_m": scale, "valid_depth_fraction": float(valid.mean()),
        "valid_depth_percentiles_m": dict(zip(["p01", "p05", "p50", "p95", "p99"], map(float, percentiles))),
        "valid_depth_min_m": float(depth_m[valid].min()),
        "valid_depth_max_m": float(depth_m[valid].max()),
        "timestamp_domains_match": same_domain,
        "color_minus_depth_timestamp_ms": float(meta["color_timestamp_ms"] - meta["depth_timestamp_ms"]) if same_domain else None,
        "K": color["K"], "distortion_coeffs": color["coeffs"],
        "depth_to_color_translation_m": meta["depth_to_color"]["translation_m"],
        "camera_to_robot_extrinsics_available": meta.get("camera_to_robot_extrinsics") is not None,
        "limits": "Single saved sample; no metric accuracy, frame-drop or camera-to-robot calibration validation.",
    }
    # NaN仅用于展示缺测区域；保存的原始深度文件保持不变。
    display = np.where(valid, depth_m, np.nan)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("black")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("D435i RGB (640 x 480)")
    image = axes[1].imshow(display, cmap=cmap, vmin=0, vmax=float(percentiles[-1]))
    axes[1].set_title(f"Aligned depth: {valid.mean():.3%} nonzero")
    fig.colorbar(image, ax=axes[1], label="Depth (m); display clipped at valid p99")
    for ax in axes:
        ax.set_xlabel("u (pixel)")
        ax.set_ylabel("v (pixel)")
    fig.savefig(sample / "rgbd_preview.png", dpi=150)
    plt.close(fig)
    (sample / "analysis_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.sample), indent=2))
