"""Composite a generated architecture with exact real-data result panels."""

import argparse
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from PIL import Image

from nets import nn
from utils import util
from utils.ihc_dataset import (
    IHCHierarchicalDataset,
    PDL1_CLASS_COLORS,
    PDL1_CLASS_NAMES,
    discover_records,
    split_records_by_slide_balanced,
)


DEFAULT_BASE_IMAGE = Path("figures/pdl1_hierarchy_generated_base.png")
DEFAULT_OUTPUT_STEM = Path("figures/pdl1_hierarchical_architecture_paper")
CHECKPOINT = Path(
    "../../model/mediautoAI_PDL1_Stomach/"
    "pdl1_hierarchical_eio_yolov11/best_model.pt"
)
DATA_ROOT = Path("../../data/mediautoAI_PDL1_Stomach")
SAMPLE_INDEX = 14
CONFIDENCE_THRESHOLD = 0.10
FINAL_WIDTH = 3072
GENERATED_BASE_HEIGHT = 2048
DEFAULT_FINAL_HEIGHT = 2048
DEFAULT_BOTTOM_Y = 1156


def scatter_cells(axis, classes, centers):
    for class_id in range(7):
        mask = classes == class_id
        if np.any(mask):
            axis.scatter(
                centers[mask, 0], centers[mask, 1],
                s=22, facecolors="none",
                edgecolors=PDL1_CLASS_COLORS[class_id],
                linewidths=1.15, alpha=0.97,
            )


def render_exact_bottom(width, height):
    records = discover_records(DATA_ROOT)
    _, validation_records = split_records_by_slide_balanced(
        records, val_slide_count=3
    )
    dataset = IHCHierarchicalDataset(
        validation_records, input_size=512, resize_size=1024,
        augment=False, random_crop=False, cache_images=True, four_crop=True,
    )
    image, classes, boxes, _ = dataset[SAMPLE_INDEX]
    device = torch.device(
        "cuda:1" if torch.cuda.device_count() > 1
        else "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = nn.yolo_v11_m_pdl1_hierarchical().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        model_input = image.unsqueeze(0).to(device).float() / 255.0
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            prediction = model(model_input)
        detections = util.non_max_suppression(
            prediction, confidence_threshold=CONFIDENCE_THRESHOLD,
            iou_threshold=0.45, class_agnostic=True,
        )[0].float().cpu()

    image_array = image.permute(1, 2, 0).numpy()
    gt_classes = classes.long().numpy()
    gt_centers = boxes[:, :2].numpy() * 512
    pred_classes = detections[:, 5].long().numpy()
    pred_centers = np.column_stack((
        (detections[:, 0].numpy() + detections[:, 2].numpy()) / 2,
        (detections[:, 1].numpy() + detections[:, 3].numpy()) / 2,
    ))

    dpi = 200
    figure, axes = plt.subplots(
        1, 3, figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white"
    )
    titles = (
        "(a)  Actual 512×512 validation tile",
        f"(b)  Ground truth: {len(gt_classes)} annotated cells",
        "(c)  Actual best-checkpoint prediction\n"
        f"{len(pred_classes)} detections at confidence ≥ 0.10 · best epoch "
        f"{checkpoint['epoch'] + 1}",
    )
    for axis, title in zip(axes, titles):
        axis.imshow(image_array)
        axis.set_title(title, fontsize=12.5, fontweight="bold", pad=5)
        axis.axis("off")
    scatter_cells(axes[1], gt_classes, gt_centers)
    scatter_cells(axes[2], pred_classes, pred_centers)
    legend = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor="none",
            markeredgecolor=PDL1_CLASS_COLORS[class_id], markeredgewidth=1.5,
            label=PDL1_CLASS_NAMES[class_id], markersize=6.5,
        )
        for class_id in range(7)
    ]
    figure.legend(
        handles=legend, loc="lower center", ncol=7, frameon=False,
        bbox_to_anchor=(0.5, 0.006), fontsize=8.7,
    )
    figure.subplots_adjust(
        left=0.012, right=0.988, top=0.87, bottom=0.105, wspace=0.025
    )
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, facecolor="white")
    plt.close(figure)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE_IMAGE)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--bottom-y", type=int, default=DEFAULT_BOTTOM_Y)
    parser.add_argument(
        "--final-height", type=int, default=DEFAULT_FINAL_HEIGHT
    )
    arguments = parser.parse_args()
    if not 0 < arguments.bottom_y < arguments.final_height:
        raise ValueError("bottom-y must fall inside the final image")

    with Image.open(arguments.base) as source:
        generated = source.convert("RGB").resize(
            (FINAL_WIDTH, GENERATED_BASE_HEIGHT), Image.Resampling.LANCZOS
        )
    canvas = Image.new(
        "RGB", (FINAL_WIDTH, arguments.final_height), color="white"
    )
    architecture_top = generated.crop(
        (0, 0, FINAL_WIDTH, arguments.bottom_y)
    )
    canvas.paste(architecture_top, (0, 0))
    bottom = render_exact_bottom(
        FINAL_WIDTH, arguments.final_height - arguments.bottom_y
    )
    canvas.paste(bottom, (0, arguments.bottom_y))
    output_png = arguments.output_stem.with_suffix(".png")
    output_tiff = arguments.output_stem.with_suffix(".tiff")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png, dpi=(300, 300), optimize=True)
    canvas.save(output_tiff, dpi=(300, 300), compression="tiff_lzw")
    print(output_png.resolve())
    print(output_tiff.resolve())


if __name__ == "__main__":
    main()
