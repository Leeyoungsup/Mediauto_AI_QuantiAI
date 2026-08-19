"""Create exact image/GT/prediction panels for the PD-L1 paper figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from nets import nn
from utils import util
from utils.ihc_dataset import (
    IHCHierarchicalDataset,
    PDL1_CLASS_COLORS,
    PDL1_CLASS_NAMES,
    discover_records,
    split_records_by_slide_balanced,
)


DATA_ROOT = Path("../../data/mediautoAI_PDL1_Stomach")
CHECKPOINT = Path(
    "../../model/mediautoAI_PDL1_Stomach/"
    "pdl1_hierarchical_eio_yolov11/best_model.pt"
)
OUTPUT = Path("figures/pdl1_hierarchy_actual_panels.png")
SAMPLE_INDEX = 14
CONFIDENCE_THRESHOLD = 0.10


def scatter_cells(axis, classes, centers, size=16):
    for class_id in range(7):
        mask = classes == class_id
        if np.any(mask):
            axis.scatter(
                centers[mask, 0], centers[mask, 1],
                s=size, facecolors="none", edgecolors=PDL1_CLASS_COLORS[class_id],
                linewidths=0.9, alpha=0.95,
            )


def main():
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
            components = model(model_input, return_components=True)
        detections = util.non_max_suppression(
            components["predictions"],
            confidence_threshold=CONFIDENCE_THRESHOLD,
            iou_threshold=0.45,
            class_agnostic=True,
        )[0].float().cpu()

    image_array = image.permute(1, 2, 0).numpy()
    gt_classes = classes.long().numpy()
    gt_centers = boxes[:, :2].numpy() * 512
    if len(detections):
        pred_classes = detections[:, 5].long().numpy()
        pred_centers = np.column_stack((
            (detections[:, 0].numpy() + detections[:, 2].numpy()) / 2,
            (detections[:, 1].numpy() + detections[:, 3].numpy()) / 2,
        ))
    else:
        pred_classes = np.empty(0, dtype=np.int64)
        pred_centers = np.empty((0, 2), dtype=np.float32)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5.6), facecolor="white")
    titles = (
        "Actual validation tile\n512 × 512 H&E/PD-L1 patch",
        f"Ground truth\n{len(gt_classes)} annotated cells",
        f"Best checkpoint prediction\n{len(pred_classes)} detections (conf ≥ 0.10)",
    )
    for axis, title in zip(axes, titles):
        axis.imshow(image_array)
        axis.set_title(title, fontsize=13, fontweight="bold", pad=9)
        axis.axis("off")
    scatter_cells(axes[1], gt_classes, gt_centers)
    scatter_cells(axes[2], pred_classes, pred_centers)

    legend = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor="none",
            markeredgecolor=PDL1_CLASS_COLORS[class_id], markeredgewidth=1.5,
            label=PDL1_CLASS_NAMES[class_id], markersize=7,
        )
        for class_id in range(7)
    ]
    figure.legend(
        handles=legend, loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, 0.005), fontsize=10,
    )
    figure.suptitle(
        f"Real validation example used in the architecture figure · "
        f"best epoch {checkpoint['epoch'] + 1}",
        fontsize=15, fontweight="bold", y=0.985,
    )
    figure.subplots_adjust(left=0.015, right=0.985, top=0.88, bottom=0.15, wspace=0.035)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=240, facecolor="white")
    plt.close(figure)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
