"""Folder-wide flat-five inference and patch-level metrics."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from nets import nn
from utils import util
from utils.ihc_dataset import IMAGE_SUFFIXES
from utils.flat5_eval_dataset import Flat5EvaluationDataset, FLAT5_CLASS_NAMES
from utils.rectangle_metrics import (
    clip_boxes, rectangle_dice, average_precision_50,
    cps_from_flat5_counts, cps_grade, paired_score_correlation,
)


def load_test_folder(folder):
    """Read every image recursively; require a sibling JSON, including empty cells lists."""
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    images = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f'No images in {folder}')
    missing = [str(p) for p in images if not p.with_suffix('.json').is_file()]
    if missing:
        raise FileNotFoundError(f'Images without GT JSON (not silently treated as background): {missing}')
    records = [(p, p.with_suffix('.json')) for p in images]
    return Flat5EvaluationDataset(records, input_size=512, resize_size=1024,
        augment=False, random_crop=False, four_crop=True, cache_images=False)


def predict_patches(dataset, checkpoint_path, device, ap_confidence=0.001, nms_iou=0.45):
    """Resize to 1024², infer all four fixed 512² quadrants, including empty-GT crops."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint.get('num_classes') != 5 or checkpoint.get('architecture') != 'pdl1_flat_5class':
        raise ValueError('Expected a flat-five checkpoint')
    names = checkpoint.get('class_names', checkpoint.get('params', {}).get('names'))
    if names is None or {int(k): v for k, v in names.items()} != FLAT5_CLASS_NAMES:
        raise ValueError('Checkpoint class order differs')
    model = nn.yolo_v11_m(num_classes=5)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    del checkpoint
    model = model.to(device).eval()
    results = []
    with torch.inference_mode():
        for index in tqdm(range(len(dataset)), desc=Path(checkpoint_path).parent.name):
            image, labels, boxes, _ = dataset[index]
            xywh = boxes.numpy() * 512
            gt, keep = clip_boxes(np.concatenate(
                [xywh[:, :2]-xywh[:, 2:]/2, xywh[:, :2]+xywh[:, 2:]/2], axis=1), 512, 512)
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                raw = model(image.unsqueeze(0).to(device).float()/255)
            pred = util.non_max_suppression(raw.float(), confidence_threshold=ap_confidence,
                iou_threshold=nms_iou, class_agnostic=True)[0].cpu().numpy()
            pred_boxes, valid = clip_boxes(pred[:, :4], 512, 512)
            source = str(dataset.records[index//4][0])
            results.append({'patch': f'{source}::q{index%4}', 'source_image': source,
                'quadrant': index%4, 'gt_boxes': gt[keep],
                'gt_labels': labels.numpy().astype(int)[keep], 'pred_boxes': pred_boxes[valid],
                'pred_labels': pred[valid, 5].astype(int), 'pred_scores': pred[valid, 4]})
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return results


def calculate_dice(records, confidence=0.1):
    """Class-agnostic filled-rectangle union Dice; pooled across all patches."""
    rows = []
    for r in records:
        selected = np.asarray(r['pred_scores']) > confidence
        rows.append({'patch': r['patch'], **rectangle_dice(
            r['gt_boxes'], np.asarray(r['pred_boxes'])[selected], 512, 512)})
    table = pd.DataFrame(rows)
    denominator = table.gt_area.sum() + table.pred_area.sum()
    score = 2*table.intersection.sum()/denominator if denominator else float('nan')
    return float(score), table


def calculate_map(records):
    """Class-aware mAP@0.5 over all five classes, including other."""
    rows, score = average_precision_50(records, FLAT5_CLASS_NAMES)
    table = pd.DataFrame(rows)
    table['class_name'] = table.class_id.map(FLAT5_CLASS_NAMES)
    return score, table


def calculate_pcc(records, confidence=0.1, thresholds=(1, 5, 10)):
    """Pearson correlation of patch CPS grades; report undefined CPS pairs explicitly."""
    rows = []
    for r in records:
        gt_counts = np.bincount(np.asarray(r['gt_labels'], dtype=int), minlength=5)
        selected = np.asarray(r['pred_scores']) > confidence
        pred_counts = np.bincount(np.asarray(r['pred_labels'], dtype=int)[selected], minlength=5)
        gt = cps_from_flat5_counts(gt_counts)['cps']
        pred = cps_from_flat5_counts(pred_counts)['cps']
        rows.append({'patch': r['patch'], 'gt_count': int(gt_counts.sum()),
            'pred_count': int(pred_counts.sum()), 'gt_cps': gt, 'pred_cps': pred,
            'gt_grade': cps_grade(gt, thresholds), 'pred_grade': cps_grade(pred, thresholds)})
    table = pd.DataFrame(rows)
    stats = paired_score_correlation(table.gt_grade, table.pred_grade)
    valid = np.isfinite(table.gt_grade) & np.isfinite(table.pred_grade)
    stats['grade_accuracy'] = float((table.loc[valid, 'gt_grade'] == table.loc[valid, 'pred_grade']).mean()) if valid.any() else float('nan')
    return stats, table
