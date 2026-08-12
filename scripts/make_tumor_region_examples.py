import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nets import nn
from utils import util


CLASS_NAMES = {
    0: "class0",
    1: "class1",
    2: "class2",
    3: "class3",
    4: "other",
}


def load_model(checkpoint_path, num_classes, device):
    model = nn.yolo_v11_m(num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def load_image(path, input_size):
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise FileNotFoundError(path)
    image_bgr = cv2.resize(image_bgr, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).contiguous().float() / 255.0
    return image_rgb, tensor.unsqueeze(0)


def predict_points(model, image_tensor, device, conf_threshold, nms_iou):
    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        detections = util.non_max_suppression(
            outputs,
            confidence_threshold=conf_threshold,
            iou_threshold=nms_iou,
            class_agnostic=False,
        )[0]

    if detections.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32)

    detections = detections.detach().cpu().numpy()
    x1, y1, x2, y2 = detections[:, 0], detections[:, 1], detections[:, 2], detections[:, 3]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    conf = detections[:, 4]
    cls = detections[:, 5]
    return np.stack([cx, cy, cls, conf], axis=1).astype(np.float32)


def label_points(label_path, input_size):
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    with open(label_path) as f:
        labels = json.load(f)
    points = []
    for item in labels:
        points.append(
            [
                float(item["cx"]) * input_size,
                float(item["cy"]) * input_size,
                float(item["class_id"]),
                1.0,
            ]
        )
    return np.array(points, dtype=np.float32) if points else np.zeros((0, 4), dtype=np.float32)


def make_density(points, class_ids, shape, sigma):
    density = np.zeros(shape, dtype=np.float32)
    if len(points) == 0:
        return density

    mask = np.isin(points[:, 2].astype(np.int32), class_ids)
    for x, y, _, conf in points[mask]:
        xi = int(np.clip(round(x), 0, shape[1] - 1))
        yi = int(np.clip(round(y), 0, shape[0] - 1))
        density[yi, xi] += float(conf)

    ksize = int(max(3, round(sigma * 6) // 2 * 2 + 1))
    density = cv2.GaussianBlur(density, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    if density.max() > 0:
        density /= density.max()
    return density


def clean_mask(mask, min_area):
    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((9, 9), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    cleaned = np.zeros_like(mask_u8)
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == idx] = 255
    return cleaned


def smooth_filled_regions(tumor_region, support):
    region = (tumor_region > 0).astype(np.uint8) * 255
    region = cv2.medianBlur(region, 21)

    kernel = np.ones((21, 21), np.uint8)
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, kernel, iterations=1)
    region = cv2.morphologyEx(region, cv2.MORPH_OPEN, kernel, iterations=1)

    return ((region > 0) & support)


def make_tissue_support(image, total_density, active_threshold):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Keep stained tissue, but remove near-white empty background.
    tissue = ((saturation > 8) | (value < 242)).astype(np.uint8) * 255
    kernel = np.ones((15, 15), np.uint8)
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, kernel, iterations=2)
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN, kernel, iterations=1)

    cell_field = (total_density > active_threshold).astype(np.uint8) * 255
    field_kernel = np.ones((41, 41), np.uint8)
    cell_field = cv2.dilate(cell_field, field_kernel, iterations=1)
    cell_field = cv2.morphologyEx(cell_field, cv2.MORPH_CLOSE, field_kernel, iterations=1)

    return ((tissue > 0) & (cell_field > 0))


def overlay_regions(image, tumor_mask, non_tumor_mask):
    overlay = image.copy()
    tumor_color = np.array([255, 40, 40], dtype=np.uint8)
    non_tumor_color = np.array([30, 180, 255], dtype=np.uint8)

    for mask, color in [(tumor_mask, tumor_color), (non_tumor_mask, non_tumor_color)]:
        alpha = (mask > 0).astype(np.float32)[:, :, None] * 0.25
        overlay = (overlay * (1.0 - alpha) + color * alpha).astype(np.uint8)

    contours_t, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_n, _ = cv2.findContours(non_tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours_t, -1, (255, 0, 0), 2)
    cv2.drawContours(overlay, contours_n, -1, (0, 140, 255), 2)
    return overlay


def save_figure(image, points, tumor_density, other_density, tumor_score, tumor_mask, non_tumor_mask, out_path, title):
    overlay = overlay_regions(image, tumor_mask, non_tumor_mask)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    axes = axes.ravel()
    axes[0].imshow(image)
    axes[0].set_title("Patch")

    axes[1].imshow(image)
    tumor = points[np.isin(points[:, 2].astype(np.int32), [0, 1, 2, 3])] if len(points) else np.zeros((0, 4))
    other = points[points[:, 2].astype(np.int32) == 4] if len(points) else np.zeros((0, 4))
    if len(tumor):
        axes[1].scatter(tumor[:, 0], tumor[:, 1], s=7, c="#ff3030", label="class0-3", alpha=0.75)
    if len(other):
        axes[1].scatter(other[:, 0], other[:, 1], s=7, c="#1e9fff", label="other", alpha=0.75)
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].set_title(f"Points: tumor={len(tumor)}, other={len(other)}")

    axes[2].imshow(tumor_density, cmap="Reds", vmin=0, vmax=1)
    axes[2].set_title("Tumor-cell density")

    axes[3].imshow(other_density, cmap="Blues", vmin=0, vmax=1)
    axes[3].set_title("Other-cell density")

    axes[4].imshow(tumor_score, cmap="coolwarm", vmin=0, vmax=1)
    axes[4].set_title("Tumor score")

    axes[5].imshow(overlay)
    axes[5].set_title("Region overlay")

    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def choose_images(image_dir, label_dir, n):
    candidates = []
    for image_path in sorted(image_dir.glob("*.png")):
        label_path = label_dir / f"{image_path.stem}.json"
        points = label_points(label_path, 512)
        if len(points) == 0:
            continue
        tumor_count = int(np.isin(points[:, 2].astype(np.int32), [0, 1, 2, 3]).sum())
        other_count = int((points[:, 2].astype(np.int32) == 4).sum())
        if tumor_count > 8 and other_count > 8:
            candidates.append((min(tumor_count, other_count), tumor_count + other_count, image_path))
    candidates.sort(reverse=True)
    return [item[2] for item in candidates[:n]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="../../data/precise_BC_cell_scoring/her2/patch_images")
    parser.add_argument("--label-dir", default="../../data/precise_BC_cell_scoring/her2/labels")
    parser.add_argument("--checkpoint", default="../../model/precise_BC_cell_scoring/her2_yolov11/best_model.pt")
    parser.add_argument("--out-dir", default="docs/results/tumor_region_examples")
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--sigma", type=float, default=24.0)
    parser.add_argument("--min-area", type=int, default=900)
    parser.add_argument("--active-threshold", type=float, default=0.03)
    parser.add_argument("--filled-regions", action="store_true")
    parser.add_argument("--source", choices=["model", "label"], default="model")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if args.source == "model":
        model = load_model(args.checkpoint, len(CLASS_NAMES), device)

    selected = choose_images(image_dir, label_dir, args.num_images)
    if len(selected) < args.num_images:
        selected = sorted(image_dir.glob("*.png"))[: args.num_images]

    index_lines = ["# Tumor / Non-tumor Region Examples", ""]
    for idx, image_path in enumerate(selected, start=1):
        image, tensor = load_image(image_path, args.input_size)
        if model is None:
            points = label_points(label_dir / f"{image_path.stem}.json", args.input_size)
            point_source = "label"
        else:
            points = predict_points(model, tensor, device, args.conf_threshold, args.nms_iou)
            point_source = "model"

        tumor_density = make_density(points, [0, 1, 2, 3], image.shape[:2], args.sigma)
        other_density = make_density(points, [4], image.shape[:2], args.sigma)
        total_density = tumor_density + other_density
        tumor_score = tumor_density / (total_density + 1e-6)

        if args.filled_regions:
            support = make_tissue_support(image, total_density, args.active_threshold)
            tumor_region = smooth_filled_regions((tumor_score >= 0.5) & support, support)

            # In filled mode, every supported tissue pixel must belong to exactly
            # one region. Do not clean the two masks independently, since that
            # creates unlabeled gaps between tumor and non-tumor boundaries.
            tumor_mask = tumor_region.astype(np.uint8) * 255
            non_tumor_mask = (support & (~tumor_region)).astype(np.uint8) * 255
        else:
            active = total_density > 0.08
            tumor_mask = clean_mask((tumor_score >= 0.55) & active, args.min_area)
            non_tumor_mask = clean_mask((tumor_score <= 0.35) & active, args.min_area)

        out_name = f"example_{idx}_{image_path.stem}.png"
        out_path = out_dir / out_name
        title = f"{image_path.name} ({point_source} points)"
        save_figure(image, points, tumor_density, other_density, tumor_score, tumor_mask, non_tumor_mask, out_path, title)

        index_lines.append(f"## Example {idx}")
        index_lines.append("")
        index_lines.append(f"- Patch: `{image_path.name}`")
        index_lines.append(f"- Point source: `{point_source}`")
        index_lines.append(f"- Detections/points: `{len(points)}`")
        index_lines.append("")
        index_lines.append(f"![Example {idx}]({out_name})")
        index_lines.append("")
        print(out_path)

    (out_dir / "README.md").write_text("\n".join(index_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
