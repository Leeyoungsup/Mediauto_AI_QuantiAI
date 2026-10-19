"""Metric primitives and disk-backed prediction storage for detection notebooks."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

# 박스 및 사각형 Dice

def clip_boxes(boxes, height, width):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    if not np.isfinite(boxes).all():
        raise ValueError('Boxes must be finite')
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes, valid

def rectangle_mask(boxes, height, width):
    """Union of rectangles, floor left/top and ceil right/bottom, exclusive end."""
    boxes, valid = clip_boxes(boxes, height, width)
    mask = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in boxes[valid]:
        mask[int(np.floor(y1)):int(np.ceil(y2)),
             int(np.floor(x1)):int(np.ceil(x2))] = True
    return mask

def rectangle_dice(gt_boxes, pred_boxes, height, width):
    gt = rectangle_mask(gt_boxes, height, width)
    pred = rectangle_mask(pred_boxes, height, width)
    intersection = int(np.count_nonzero(gt & pred))
    gt_area, pred_area = int(gt.sum()), int(pred.sum())
    denominator = gt_area + pred_area
    return {
        'dice': 2 * intersection / denominator if denominator else float('nan'),
        'intersection': intersection, 'gt_area': gt_area, 'pred_area': pred_area,
    }


# AP@0.5

def box_iou(box, boxes):
    boxes = np.asarray(boxes).reshape(-1, 4)
    intersection = np.maximum(0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2])).prod(axis=1)
    area = np.maximum(0, box[2:] - box[:2]).prod()
    areas = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    return intersection / np.maximum(area + areas - intersection, 1e-12)

def average_precision_50(records, class_ids):
    """101-recall-point interpolated AP, IoU >= .50, one-to-one per-image/class.

    Records contain gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores.
    No crowd/ignore flags or maximum-detection truncation. Absent-GT classes
    have NaN AP and are excluded from mAP. A class with GT but no predictions
    has AP zero. Call after low-confidence-threshold NMS, not display filtering.
    """
    rows = []
    for cls in class_ids:
        ground_truth = [np.asarray(r['gt_boxes']).reshape(-1, 4)[np.asarray(r['gt_labels']) == cls] for r in records]
        matched = [np.zeros(len(boxes), dtype=bool) for boxes in ground_truth]
        n_gt = sum(map(len, ground_truth))
        predictions = []
        for image_index, record in enumerate(records):
            for box, label, score in zip(record['pred_boxes'], record['pred_labels'], record['pred_scores']):
                if label == cls:
                    predictions.append((float(score), image_index, np.asarray(box)))
        predictions.sort(key=lambda row: -row[0])
        tp = np.zeros(len(predictions), dtype=np.float64)
        for i, (_, image_index, box) in enumerate(predictions):
            available = np.flatnonzero(~matched[image_index])
            if available.size:
                ious = box_iou(box, ground_truth[image_index][available])
                best = int(np.argmax(ious))
                if ious[best] >= .5:
                    tp[i] = 1
                    matched[image_index][available[best]] = True
        if n_gt == 0:
            ap = float('nan')
        elif not len(predictions):
            ap = 0.0
        else:
            recall = tp.cumsum() / n_gt
            precision = tp.cumsum() / np.arange(1, len(tp) + 1)
            envelope = np.maximum.accumulate(precision[::-1])[::-1]
            ap = float(np.mean([
                envelope[np.searchsorted(recall, threshold, side='left')]
                if np.searchsorted(recall, threshold, side='left') < len(envelope) else 0.0
                for threshold in np.linspace(0, 1, 101)
            ]))
        rows.append({'class_id': cls, 'ap50': ap, 'gt_count': n_gt,
                     'ap_prediction_count': len(predictions), 'tp50': int(tp.sum()),
                     'fp50': len(predictions) - int(tp.sum())})
    values = [row['ap50'] for row in rows if np.isfinite(row['ap50'])]
    return rows, float(np.mean(values)) if values else float('nan')


# CPS Grade 및 PCC

def count_correlation(gt_counts, pred_counts):
    gt, pred = np.asarray(gt_counts, dtype=float), np.asarray(pred_counts, dtype=float)
    if gt.shape != pred.shape or gt.ndim != 1 or not np.isfinite(gt).all() or not np.isfinite(pred).all():
        raise ValueError('Counts must be equal-length finite one-dimensional arrays')
    if len(gt) < 2:
        pcc, reason = float('nan'), 'fewer_than_two_samples'
    elif np.ptp(gt) == 0 or np.ptp(pred) == 0:
        pcc, reason = float('nan'), 'constant_counts'
    else:
        pcc, reason = float(pearsonr(gt, pred)[0]), ''
    return {'pcc': pcc, 'pcc_undefined_reason': reason, 'n_samples': len(gt),
            'count_mae': float(np.abs(pred-gt).mean()) if len(gt) else float('nan'),
            'count_bias': float((pred-gt).mean()) if len(gt) else float('nan')}

def cps_from_counts(counts):
    """Label-derived CPS proxy: 100*(positive epithelial+immune)/all epithelial.

    Counts follow Neg Epi, Neg Immune, Pos Epi, Pos Immune, other order.
    Assumes epithelial labels denote viable tumor and positive immune labels
    denote eligible tumor-associated immune cells. No tumor gives NaN, not zero.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.shape != (5,) or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('Expected five finite nonnegative class counts')
    tumor = float(counts[0] + counts[2])
    positive = float(counts[2] + counts[3])
    raw = 100.0 * positive / tumor if tumor else float('nan')
    return {'cps': min(raw, 100.0) if tumor else float('nan'),
            'cps_raw': raw, 'tumor_count': tumor, 'positive_count': positive}

def cps_grade(cps, thresholds):
    """Zero-based ordinal grade; equality at a cutoff enters the upper grade."""
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or not len(thresholds) or not np.isfinite(thresholds).all() or np.any(np.diff(thresholds) <= 0):
        raise ValueError('Grade thresholds must be a strictly increasing finite list')
    return int(np.searchsorted(thresholds, cps, side='right')) if np.isfinite(cps) else float('nan')

def paired_score_correlation(gt, pred):
    """PCC over valid pairs, explicitly reporting excluded pairs and coverage."""
    gt, pred = np.asarray(gt, dtype=float), np.asarray(pred, dtype=float)
    if gt.ndim != 1 or gt.shape != pred.shape:
        raise ValueError('Expected equal-length one-dimensional score arrays')
    valid = np.isfinite(gt) & np.isfinite(pred)
    result = count_correlation(gt[valid], pred[valid])
    return {'pcc': result['pcc'], 'pcc_undefined_reason': result['pcc_undefined_reason'],
            'n_pairs': len(gt), 'n_valid_pairs': int(valid.sum()),
            'n_invalid_gt': int((~np.isfinite(gt)).sum()),
            'n_invalid_pred': int((~np.isfinite(pred)).sum()),
            'valid_pair_fraction': float(valid.mean()) if len(gt) else float('nan'),
            'mae': result['count_mae'], 'bias': result['count_bias']}


class PredictionStore:
    """Re-iterable per-patch NPZ files: keep predictions off the Python heap."""
    def __init__(self, directory, create=False):
        self.directory = Path(directory)
        if create:
            self.directory.mkdir(parents=True, exist_ok=False)
            self.count = 0
        else:
            self.count = len(list(self.directory.glob('patch_*.npz')))

    def append(self, record):
        np.savez(self.directory / f'patch_{self.count:08d}.npz', **record)
        self.count += 1

    def __len__(self):
        return self.count

    def __iter__(self):
        for index in range(self.count):
            with np.load(self.directory / f'patch_{index:08d}.npz', allow_pickle=False) as data:
                yield {key: data[key].item() if data[key].ndim == 0 else data[key]
                       for key in data.files}


def save_predictions_json(records, path):
    """Optional JSON export, one patch at a time without a full JSON string."""
    with Path(path).open('w', encoding='utf-8') as stream:
        stream.write('[')
        for index, record in enumerate(records):
            if index:
                stream.write(',')
            json.dump(record, stream, default=lambda value: value.tolist())
        stream.write(']')
