"""Build the Tumor/Other checkpoint evaluation notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Precise_PDL1_ST_tumor_other_test.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python (urban)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.11"},
}

nb["cells"] = [
    markdown(
        r"""
# PD-L1 Stomach — Tumor / Other Binary Model Test

`Precise_PDL1_ST_tumor_other_train.ipynb`에서 학습한 일반 2-class YOLOv11-m 모델을 평가합니다.

- **Tumor (0)**: Negative Epithelial + Positive Epithelial
- **Other (1)**: Lymphocyte + Macrophage + `other`
- 1024×1024 원본 패치 하나를 겹치지 않는 512×512 타일 4장으로 나누어 모두 평가합니다.
- augmentation과 random crop은 사용하지 않습니다.
- 중심점 거리 16 px 이내의 예측과 GT를 Hungarian matching합니다.
- confidence 0.10의 고정 결과와 threshold sweep의 사후 최적 결과를 함께 출력합니다.

> **중요:** 기본 설정은 학습 때 모델 선택에 사용한 3개 validation slide를 그대로 복원합니다. 따라서 이것은 독립적인 외부 test 성능이 아니라 **development hold-out 재평가**입니다. 논문 최종 test 수치에는 학습/모델 선택에 전혀 사용하지 않은 별도 slide가 필요합니다.
"""
    ),
    code(
        r"""
import json
import os
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from IPython.display import display
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

from nets import nn
from utils import util
from utils.ihc_dataset import (
    IHCHierarchicalDataset,
    count_record_classes,
    discover_records,
    slide_id_from_record,
    split_records_by_slide_balanced,
)

warnings.filterwarnings('ignore')
random.seed(242)
np.random.seed(242)
torch.manual_seed(242)

# ── 실행 설정 ─────────────────────────────────────────────────────
GPU_ID = 1
DATA_ROOT = Path('../../data/mediautoAI_PDL1_Stomach/')
CHECKPOINT_PATH = Path(
    '../../model/mediautoAI_PDL1_Stomach/'
    'pdl1_tumor_other_yolov11/best_model.pt'
)
OUTPUT_DIR = CHECKPOINT_PATH.parent / 'test_results'

INPUT_SIZE = 512
RESIZE_SIZE = 1024
BATCH_SIZE = 16
VALIDATION_SLIDE_COUNT = 3
DISTANCE_THRESHOLD_PX = 16.0
NMS_IOU_THRESHOLD = 0.45
MIN_CACHE_CONFIDENCE = 0.01
FIXED_CONFIDENCE = 0.10

# None이면 train 노트북과 동일한 3개 validation slide를 자동 복원합니다.
# 독립 test slide 폴더명이 따로 있다면 문자열 목록으로 지정할 수 있습니다.
# 단, 기존 학습에 포함된 slide를 지정하면 독립 test가 아닙니다.
TEST_SLIDE_IDS = None

if torch.cuda.is_available():
    if GPU_ID >= torch.cuda.device_count():
        raise RuntimeError(
            f'GPU_ID={GPU_ID}, available GPUs={torch.cuda.device_count()}'
        )
    torch.cuda.set_device(GPU_ID)
    device = torch.device(f'cuda:{GPU_ID}')
    print(f'device: {device} | {torch.cuda.get_device_name(GPU_ID)}')
else:
    device = torch.device('cpu')
    print('device: cpu')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BINARY_CLASS_NAMES = {0: 'Tumor (Epithelial)', 1: 'Other (Immune + other)'}
BINARY_CLASS_COLORS = {0: '#d62728', 1: '#7f7f7f'}
SHORT_NAMES = ['Tumor', 'Other']

with open('utils/detail_args.yaml', errors='ignore') as file:
    params = yaml.safe_load(file)
params['names'] = BINARY_CLASS_NAMES
"""
    ),
    markdown("## 1. 고정 test 타일 구성"),
    code(
        r"""
class TumorOtherDataset(torch.utils.data.Dataset):
    # 지원되는 7개 PD-L1 라벨을 Tumor/Other로 remap합니다.

    class_names = BINARY_CLASS_NAMES
    class_colors = BINARY_CLASS_COLORS

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, classes, boxes, indices = self.base_dataset[index]
        is_epithelial = (classes == 0) | (classes == 3)
        binary_classes = torch.where(
            is_epithelial, torch.zeros_like(classes), torch.ones_like(classes)
        )
        return image, binary_classes, boxes, indices

    @property
    def cache_memory_bytes(self):
        return self.base_dataset.cache_memory_bytes


def collate_fn(batch):
    return IHCHierarchicalDataset.collate_fn(batch)


all_records = discover_records(DATA_ROOT)
train_records, development_records = split_records_by_slide_balanced(
    all_records, val_slide_count=VALIDATION_SLIDE_COUNT
)
train_slide_ids = {slide_id_from_record(record) for record in train_records}
development_slide_ids = {
    slide_id_from_record(record) for record in development_records
}
assert train_slide_ids.isdisjoint(development_slide_ids)

if TEST_SLIDE_IDS is None:
    test_records = development_records
    evaluation_kind = 'development hold-out (model selection에 사용됨)'
else:
    requested_slides = set(TEST_SLIDE_IDS)
    known_slides = {slide_id_from_record(record) for record in all_records}
    missing_slides = requested_slides - known_slides
    if missing_slides:
        raise ValueError(f'존재하지 않는 TEST_SLIDE_IDS: {sorted(missing_slides)}')
    test_records = [
        record for record in all_records
        if slide_id_from_record(record) in requested_slides
    ]
    if not test_records:
        raise ValueError('TEST_SLIDE_IDS에 해당하는 image/JSON pair가 없습니다.')
    overlap = requested_slides & train_slide_ids
    evaluation_kind = (
        '주의: 기존 training slide 포함' if overlap else '별도 지정 slide'
    )

test_slide_ids = sorted({slide_id_from_record(r) for r in test_records})
test_base = IHCHierarchicalDataset(
    test_records,
    input_size=INPUT_SIZE,
    resize_size=RESIZE_SIZE,
    augment=False,
    random_crop=False,
    cache_images=True,
    four_crop=True,
)
test_dataset = TumorOtherDataset(test_base)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn,
    drop_last=False,
    pin_memory=(device.type == 'cuda'),
    persistent_workers=True,
)

source_counts = count_record_classes(test_records).astype(np.int64)
binary_counts = np.array([
    source_counts[0] + source_counts[3],
    source_counts[1] + source_counts[2] + source_counts[4]
    + source_counts[5] + source_counts[6],
])

print(f'Evaluation kind       : {evaluation_kind}')
print(f'Slides                : {len(test_slide_ids)}')
for slide_id in test_slide_ids:
    patch_count = sum(slide_id_from_record(r) == slide_id for r in test_records)
    print(f'  - {slide_id}: {patch_count} source patches / {patch_count * 4} tiles')
print(f'Source patches / tiles: {len(test_records)} / {len(test_dataset)}')
print(f'Tumor / Other labels  : {binary_counts[0]:,} / {binary_counts[1]:,}')
print(f'1024 RAM cache        : {test_dataset.cache_memory_bytes / 1024**2:.1f} MiB')
"""
    ),
    markdown("## 2. Best checkpoint 로드"),
    code(
        r"""
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f'Checkpoint가 없습니다: {CHECKPOINT_PATH}')

checkpoint = torch.load(
    CHECKPOINT_PATH, map_location=device, weights_only=False
)
if checkpoint.get('architecture') != 'pdl1_tumor_other_flat':
    raise ValueError(
        f"Tumor/Other checkpoint가 아닙니다: {checkpoint.get('architecture')}"
    )
if int(checkpoint.get('num_classes', 2)) != 2:
    raise ValueError('Checkpoint num_classes가 2가 아닙니다.')

model = nn.yolo_v11_m(num_classes=2).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

stored_metrics = {
    key: checkpoint.get(key)
    for key in (
        'epoch', 'detection_recall', 'classification_accuracy',
        'macro_precision', 'macro_recall', 'macro_f1',
        'validation_score', 'best_validation_score',
    )
}
print(f'Checkpoint: {CHECKPOINT_PATH}')
print(f"Epoch     : {checkpoint.get('epoch', -1) + 1}")
display(pd.DataFrame([stored_metrics]))
"""
    ),
    markdown(
        r"""
## 3. 한 번만 추론하여 prediction cache 생성

최저 confidence 0.01에서 class-agnostic NMS를 한 결과를 CPU에 저장합니다. 이후 threshold sweep은 모델을 반복 추론하지 않고 이 cache를 필터링합니다. 높은 confidence detection은 낮은 confidence detection보다 먼저 NMS되므로 이 방식은 각 threshold에서 NMS를 다시 실행한 것과 같습니다.
"""
    ),
    code(
        r"""
prediction_cache = []
ground_truth_cache = []
tile_slide_ids = []

sample_offset = 0
with torch.inference_mode():
    for images, targets in tqdm(test_loader, desc='Inference'):
        images = images.to(device, non_blocking=True).float() / 255.0
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            raw_predictions = model(images)
        detections = util.non_max_suppression(
            raw_predictions,
            confidence_threshold=MIN_CACHE_CONFIDENCE,
            iou_threshold=NMS_IOU_THRESHOLD,
            class_agnostic=True,
        )

        for batch_index in range(len(images)):
            mask = targets['idx'] == batch_index
            ground_truth_cache.append({
                'classes': targets['cls'][mask].long().cpu().numpy(),
                'boxes': targets['box'][mask].float().cpu().numpy(),
            })
            prediction_cache.append(detections[batch_index].float().cpu().numpy())

            dataset_index = sample_offset + batch_index
            source_index = dataset_index // 4
            tile_slide_ids.append(slide_id_from_record(test_records[source_index]))
        sample_offset += len(images)

assert len(prediction_cache) == len(test_dataset)
assert len(ground_truth_cache) == len(test_dataset)
print(f'Cached tiles: {len(prediction_cache)}')
print(
    'Cached detections: '
    f'{sum(len(item) for item in prediction_cache):,} '
    f'(confidence >= {MIN_CACHE_CONFIDENCE})'
)
"""
    ),
    markdown("## 4. Point detection + binary classification 평가 함수"),
    code(
        r"""
def match_one_tile(gt_boxes, predictions, distance_threshold_px):
    # 거리 기반 1:1 Hungarian matching 결과를 반환합니다.
    gt_centers = np.zeros((len(gt_boxes), 2), dtype=np.float32)
    if len(gt_boxes):
        gt_centers = gt_boxes[:, :2] * INPUT_SIZE

    pred_centers = np.zeros((len(predictions), 2), dtype=np.float32)
    if len(predictions):
        pred_centers[:, 0] = (predictions[:, 0] + predictions[:, 2]) / 2
        pred_centers[:, 1] = (predictions[:, 1] + predictions[:, 3]) / 2

    if not len(gt_centers) or not len(pred_centers):
        return [], set(), set()

    distances = np.linalg.norm(
        gt_centers[:, None, :] - pred_centers[None, :, :], axis=2
    )
    gt_indices, pred_indices = linear_sum_assignment(distances)
    pairs = [
        (int(g), int(p), float(distances[g, p]))
        for g, p in zip(gt_indices, pred_indices)
        if distances[g, p] <= distance_threshold_px
    ]
    return pairs, {g for g, _, _ in pairs}, {p for _, p, _ in pairs}


def evaluate_cached(confidence_threshold, selected_indices=None):
    # Detection과 end-to-end 분류 지표를 동시에 계산합니다.
    if selected_indices is None:
        selected_indices = range(len(prediction_cache))

    class_tp = np.zeros(2, dtype=np.int64)
    class_fp = np.zeros(2, dtype=np.int64)
    class_fn = np.zeros(2, dtype=np.int64)
    class_gt = np.zeros(2, dtype=np.int64)
    matched_confusion = np.zeros((2, 2), dtype=np.int64)
    end_to_end_confusion = np.zeros((3, 3), dtype=np.int64)
    total_gt = total_predictions = total_matched = 0
    distance_sum = 0.0
    per_image = []

    for tile_index in selected_indices:
        ground_truth = ground_truth_cache[tile_index]
        gt_classes = ground_truth['classes'].astype(np.int64)
        gt_boxes = ground_truth['boxes']
        cached = prediction_cache[tile_index]
        predictions = cached[cached[:, 4] >= confidence_threshold]
        pred_classes = (
            predictions[:, 5].astype(np.int64)
            if len(predictions) else np.empty(0, dtype=np.int64)
        )

        pairs, matched_gt, matched_pred = match_one_tile(
            gt_boxes, predictions, DISTANCE_THRESHOLD_PX
        )
        class_gt += np.bincount(gt_classes, minlength=2)[:2]
        total_gt += len(gt_classes)
        total_predictions += len(pred_classes)
        total_matched += len(pairs)
        distance_sum += sum(distance for _, _, distance in pairs)

        for gt_index, pred_index, _ in pairs:
            gt_class = int(gt_classes[gt_index])
            pred_class = int(pred_classes[pred_index])
            matched_confusion[gt_class, pred_class] += 1
            end_to_end_confusion[gt_class, pred_class] += 1
            if gt_class == pred_class:
                class_tp[gt_class] += 1
            else:
                class_fn[gt_class] += 1
                class_fp[pred_class] += 1

        unmatched_gt = set(range(len(gt_classes))) - matched_gt
        unmatched_pred = set(range(len(pred_classes))) - matched_pred
        for gt_index in unmatched_gt:
            gt_class = int(gt_classes[gt_index])
            class_fn[gt_class] += 1
            end_to_end_confusion[gt_class, 2] += 1
        for pred_index in unmatched_pred:
            pred_class = int(pred_classes[pred_index])
            class_fp[pred_class] += 1
            end_to_end_confusion[2, pred_class] += 1

        per_image.append({
            'tile_index': int(tile_index),
            'slide_id': tile_slide_ids[tile_index],
            'gt': int(len(gt_classes)),
            'predictions': int(len(pred_classes)),
            'matched': int(len(pairs)),
            'fp': int(len(unmatched_pred)),
            'fn_detection': int(len(unmatched_gt)),
            'class_errors': int(sum(
                gt_classes[g] != pred_classes[p] for g, p, _ in pairs
            )),
        })

    precision = np.divide(
        class_tp, class_tp + class_fp,
        out=np.zeros(2, dtype=float), where=(class_tp + class_fp) > 0,
    )
    recall = np.divide(
        class_tp, class_tp + class_fn,
        out=np.zeros(2, dtype=float), where=(class_tp + class_fn) > 0,
    )
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros(2, dtype=float), where=(precision + recall) > 0,
    )
    detection_precision = total_matched / total_predictions if total_predictions else 0.0
    detection_recall = total_matched / total_gt if total_gt else 0.0
    detection_f1 = (
        2 * detection_precision * detection_recall
        / (detection_precision + detection_recall)
        if detection_precision + detection_recall else 0.0
    )
    classification_accuracy = (
        np.trace(matched_confusion) / matched_confusion.sum()
        if matched_confusion.sum() else 0.0
    )
    conditional_recalls = np.divide(
        np.diag(matched_confusion), matched_confusion.sum(axis=1),
        out=np.zeros(2, dtype=float),
        where=matched_confusion.sum(axis=1) > 0,
    )

    return {
        'confidence': float(confidence_threshold),
        'total_gt': int(total_gt),
        'total_predictions': int(total_predictions),
        'total_matched': int(total_matched),
        'detection_precision': float(detection_precision),
        'detection_recall': float(detection_recall),
        'detection_f1': float(detection_f1),
        'classification_accuracy_matched': float(classification_accuracy),
        'matched_tumor_recall': float(conditional_recalls[0]),
        'matched_other_recall': float(conditional_recalls[1]),
        'matched_balanced_accuracy': float(conditional_recalls.mean()),
        'macro_precision': float(precision.mean()),
        'macro_recall': float(recall.mean()),
        'macro_f1': float(f1.mean()),
        'validation_score': float(0.7 * f1.mean() + 0.3 * detection_recall),
        'mean_match_distance_px': float(
            distance_sum / total_matched if total_matched else 0.0
        ),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_tp': class_tp,
        'class_fp': class_fp,
        'class_fn': class_fn,
        'class_gt': class_gt,
        'matched_confusion': matched_confusion,
        'end_to_end_confusion': end_to_end_confusion,
        'per_image': per_image,
    }
"""
    ),
    markdown("## 5. Confidence threshold sweep — 성능 상한 확인"),
    code(
        r"""
thresholds = np.array([
    0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15,
    0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
])
sweep_results = [evaluate_cached(float(value)) for value in tqdm(thresholds)]

sweep_table = pd.DataFrame([
    {key: result[key] for key in (
        'confidence', 'detection_precision', 'detection_recall', 'detection_f1',
        'classification_accuracy_matched', 'matched_balanced_accuracy',
        'macro_precision', 'macro_recall', 'macro_f1', 'validation_score',
    )}
    for result in sweep_results
])

fixed_result = min(
    sweep_results, key=lambda item: abs(item['confidence'] - FIXED_CONFIDENCE)
)
best_score_result = max(
    sweep_results, key=lambda item: item['validation_score']
)
best_f1_result = max(sweep_results, key=lambda item: item['macro_f1'])

print('고정 threshold 결과 (checkpoint 선택 때와 동일한 0.10):')
display(sweep_table[np.isclose(sweep_table.confidence, FIXED_CONFIDENCE)])
print('사후 최적 selection-score threshold (진단용, 독립 test 수치로 사용 금지):')
display(sweep_table[np.isclose(
    sweep_table.confidence, best_score_result['confidence']
)])
print('사후 최적 macro-F1 threshold (진단용, 독립 test 수치로 사용 금지):')
display(sweep_table[np.isclose(
    sweep_table.confidence, best_f1_result['confidence']
)])

display(
    sweep_table.style.format({column: '{:.4f}' for column in sweep_table.columns})
)
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

for metric, label in (
    ('detection_precision', 'Detection precision'),
    ('detection_recall', 'Detection recall'),
    ('detection_f1', 'Detection F1'),
):
    axes[0].plot(sweep_table.confidence, sweep_table[metric], marker='o', label=label)
axes[0].axvline(FIXED_CONFIDENCE, color='black', linestyle='--', label='Fixed 0.10')
axes[0].set(title='Detection vs confidence', xlabel='Confidence', ylabel='Score', ylim=(0, 1))
axes[0].grid(alpha=0.25)
axes[0].legend()

for metric, label in (
    ('macro_f1', 'End-to-end macro F1'),
    ('matched_balanced_accuracy', 'Matched balanced accuracy'),
    ('validation_score', 'Selection score'),
):
    axes[1].plot(sweep_table.confidence, sweep_table[metric], marker='o', label=label)
axes[1].axvline(FIXED_CONFIDENCE, color='black', linestyle='--', label='Fixed 0.10')
axes[1].axvline(
    best_score_result['confidence'], color='#d62728', linestyle=':',
    label='Post-hoc best score',
)
axes[1].axvline(
    best_f1_result['confidence'], color='#9467bd', linestyle='-.',
    label='Post-hoc best macro F1',
)
axes[1].set(title='Classification vs confidence', xlabel='Confidence', ylabel='Score', ylim=(0, 1))
axes[1].grid(alpha=0.25)
axes[1].legend()

fig.suptitle('Tumor / Other threshold sensitivity', fontsize=15, fontweight='bold')
plt.tight_layout()
figure_path = OUTPUT_DIR / 'threshold_sweep.png'
fig.savefig(figure_path, dpi=180, bbox_inches='tight')
plt.show()
print(f'Saved: {figure_path}')
"""
    ),
    markdown("## 6. 고정 threshold 0.10 상세 결과"),
    code(
        r"""
summary_keys = [
    'total_gt', 'total_predictions', 'total_matched',
    'detection_precision', 'detection_recall', 'detection_f1',
    'classification_accuracy_matched',
    'matched_tumor_recall', 'matched_other_recall',
    'matched_balanced_accuracy',
    'macro_precision', 'macro_recall', 'macro_f1',
    'validation_score', 'mean_match_distance_px',
]
summary_table = pd.DataFrame({
    'Metric': summary_keys,
    'Fixed 0.10': [fixed_result[key] for key in summary_keys],
    f"Best score {best_score_result['confidence']:.3f}": [
        best_score_result[key] for key in summary_keys
    ],
    f"Best macro F1 {best_f1_result['confidence']:.3f}": [
        best_f1_result[key] for key in summary_keys
    ],
})
display(summary_table)

per_class_table = pd.DataFrame({
    'Class': SHORT_NAMES,
    'GT': fixed_result['class_gt'],
    'TP': fixed_result['class_tp'],
    'FP': fixed_result['class_fp'],
    'FN': fixed_result['class_fn'],
    'Precision': fixed_result['precision'],
    'Recall': fixed_result['recall'],
    'F1': fixed_result['f1'],
})
display(per_class_table.style.format({
    'Precision': '{:.4f}', 'Recall': '{:.4f}', 'F1': '{:.4f}'
}))
"""
    ),
    code(
        r"""
matrix = fixed_result['end_to_end_confusion']
labels = ['Tumor', 'Other', 'Background']
row_sum = matrix.sum(axis=1, keepdims=True)
normalized = np.divide(
    matrix, row_sum, out=np.zeros_like(matrix, dtype=float), where=row_sum > 0
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
sns.heatmap(
    matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
    xticklabels=labels, yticklabels=labels, ax=axes[0],
)
axes[0].set(title='End-to-end confusion (count)', xlabel='Prediction', ylabel='Ground truth')
sns.heatmap(
    normalized, annot=True, fmt='.3f', cmap='Blues', vmin=0, vmax=1,
    xticklabels=labels, yticklabels=labels, ax=axes[1],
)
axes[1].set(title='End-to-end confusion (row normalized)', xlabel='Prediction', ylabel='Ground truth')
plt.tight_layout()
confusion_path = OUTPUT_DIR / 'confusion_matrix_fixed_010.png'
fig.savefig(confusion_path, dpi=180, bbox_inches='tight')
plt.show()
print(f'Saved: {confusion_path}')
"""
    ),
    markdown("## 7. Slide별 결과 — 특정 slide 의존성 확인"),
    code(
        r"""
per_slide_rows = []
for slide_id in test_slide_ids:
    indices = [
        index for index, value in enumerate(tile_slide_ids) if value == slide_id
    ]
    result = evaluate_cached(FIXED_CONFIDENCE, indices)
    per_slide_rows.append({
        'Slide': slide_id,
        'Tiles': len(indices),
        'GT': result['total_gt'],
        'Predictions': result['total_predictions'],
        'Det Precision': result['detection_precision'],
        'Det Recall': result['detection_recall'],
        'Tumor Recall': result['recall'][0],
        'Other Recall': result['recall'][1],
        'Macro F1': result['macro_f1'],
        'Matched Bal Acc': result['matched_balanced_accuracy'],
    })

per_slide_table = pd.DataFrame(per_slide_rows)
display(per_slide_table.style.format({
    column: '{:.4f}' for column in (
        'Det Precision', 'Det Recall', 'Tumor Recall', 'Other Recall',
        'Macro F1', 'Matched Bal Acc',
    )
}))
"""
    ),
    markdown("## 8. 오류가 많은 실제 타일의 GT / Prediction 비교"),
    code(
        r"""
def draw_gt(axis, image, gt):
    axis.imshow(image)
    for class_id, box in zip(gt['classes'], gt['boxes']):
        x, y = box[:2] * INPUT_SIZE
        axis.scatter(
            x, y, facecolors='none', edgecolors=BINARY_CLASS_COLORS[int(class_id)],
            s=26, linewidths=1.3,
        )


def draw_predictions(axis, image, predictions):
    axis.imshow(image)
    for x1, y1, x2, y2, confidence, class_id in predictions:
        axis.scatter(
            (x1 + x2) / 2, (y1 + y2) / 2,
            facecolors='none', edgecolors=BINARY_CLASS_COLORS[int(class_id)],
            s=26, linewidths=1.3,
        )


error_table = pd.DataFrame(fixed_result['per_image'])
error_table['total_error'] = (
    error_table.fp + error_table.fn_detection + error_table.class_errors
)
worst_indices = (
    error_table.sort_values(['total_error', 'gt'], ascending=False)
    .head(min(6, len(error_table)))['tile_index'].tolist()
)

fig, axes = plt.subplots(len(worst_indices), 2, figsize=(14, 5 * len(worst_indices)))
if len(worst_indices) == 1:
    axes = np.asarray([axes])

for row, tile_index in enumerate(worst_indices):
    tensor_image, _, _, _ = test_dataset[tile_index]
    image = tensor_image.permute(1, 2, 0).numpy()
    gt = ground_truth_cache[tile_index]
    predictions = prediction_cache[tile_index]
    predictions = predictions[predictions[:, 4] >= FIXED_CONFIDENCE]
    tile_stats = error_table.loc[error_table.tile_index == tile_index].iloc[0]

    draw_gt(axes[row, 0], image, gt)
    draw_predictions(axes[row, 1], image, predictions)
    short_slide = tile_stats.slide_id.split('__')[0]
    axes[row, 0].set_title(
        f'GT | tile {tile_index} | {short_slide} | cells={len(gt["classes"])}'
    )
    axes[row, 1].set_title(
        f'Prediction | FP={tile_stats.fp}, FN={tile_stats.fn_detection}, '
        f'class error={tile_stats.class_errors}'
    )
    axes[row, 0].axis('off')
    axes[row, 1].axis('off')

legend = [
    mpatches.Patch(color=BINARY_CLASS_COLORS[index], label=SHORT_NAMES[index])
    for index in range(2)
]
fig.legend(handles=legend, loc='lower center', ncol=2)
fig.suptitle(
    f'Worst-error tiles at confidence={FIXED_CONFIDENCE:.2f}',
    fontsize=16, fontweight='bold', y=0.995,
)
plt.tight_layout(rect=[0, 0.02, 1, 0.99])
qualitative_path = OUTPUT_DIR / 'worst_error_tiles_fixed_010.png'
fig.savefig(qualitative_path, dpi=180, bbox_inches='tight')
plt.show()
print(f'Saved: {qualitative_path}')
"""
    ),
    markdown("## 9. 결과 저장 및 해석"),
    code(
        r"""
sweep_path = OUTPUT_DIR / 'threshold_sweep.csv'
class_path = OUTPUT_DIR / 'per_class_metrics_fixed_010.csv'
slide_path = OUTPUT_DIR / 'per_slide_metrics_fixed_010.csv'
summary_path = OUTPUT_DIR / 'summary_fixed_010.json'

sweep_table.to_csv(sweep_path, index=False)
per_class_table.to_csv(class_path, index=False)
per_slide_table.to_csv(slide_path, index=False)

serializable_summary = {
    key: value for key, value in fixed_result.items()
    if key not in {
        'precision', 'recall', 'f1', 'class_tp', 'class_fp', 'class_fn',
        'class_gt', 'matched_confusion', 'end_to_end_confusion', 'per_image',
    }
}
serializable_summary.update({
    'checkpoint': str(CHECKPOINT_PATH),
    'checkpoint_epoch_one_based': int(checkpoint.get('epoch', -1) + 1),
    'evaluation_kind': evaluation_kind,
    'test_slides': test_slide_ids,
    'posthoc_best_score_confidence': float(best_score_result['confidence']),
    'posthoc_best_validation_score': float(best_score_result['validation_score']),
    'posthoc_best_f1_confidence': float(best_f1_result['confidence']),
    'posthoc_best_macro_f1': float(best_f1_result['macro_f1']),
})
summary_path.write_text(
    json.dumps(serializable_summary, ensure_ascii=False, indent=2),
    encoding='utf-8',
)

print('Saved files:')
for path in (sweep_path, class_path, slide_path, summary_path):
    print(f'  - {path}')

score_gain = (
    best_score_result['validation_score'] - fixed_result['validation_score']
)
f1_gain = best_f1_result['macro_f1'] - fixed_result['macro_f1']
print('\n진단 요약')
print(f'  Fixed confidence              : {FIXED_CONFIDENCE:.3f}')
print(
    f"  Best selection-score threshold: "
    f"{best_score_result['confidence']:.3f}"
)
print(
    f"  Best macro-F1 threshold       : "
    f"{best_f1_result['confidence']:.3f}"
)
print(f'  Threshold 최적화 score gain   : {score_gain:+.4f}')
print(f'  Threshold 최적화 macro-F1 gain: {f1_gain:+.4f}')
max_gain = max(score_gain, f1_gain)
if max_gain < 0.01:
    print('  해석: confidence 조정만으로 얻을 수 있는 개선은 매우 작습니다.')
elif max_gain < 0.03:
    print('  해석: confidence 조정으로 소폭 개선 가능하지만 큰 상승은 어렵습니다.')
else:
    print('  해석: threshold calibration으로 의미 있는 개선 여지가 있습니다.')
print(
    '  주의: post-hoc best는 같은 development hold-out에서 선택했으므로 '
    '독립 test 성능으로 보고하면 안 됩니다.'
)
"""
    ),
]

nbf.validate(nb)
nbf.write(nb, OUTPUT)
print(OUTPUT)
