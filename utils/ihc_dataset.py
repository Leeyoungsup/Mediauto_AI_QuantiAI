"""Reusable lazy-loading dataset for IHC cell detection experiments."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def slide_id_from_stem(stem: str) -> str:
    """Return a slide-level group id from ``<slide>.<ext>_<x>_<y>`` names."""
    match = re.match(r'^(.*\.(?:svs|tif|tiff|ndpi))_\d+_\d+$', stem, flags=re.IGNORECASE)
    return match.group(1) if match else stem.rsplit('_', 2)[0]


def discover_records(image_dir, label_dir):
    image_dir, label_dir = Path(image_dir), Path(label_dir)
    records = []
    for label_path in sorted(label_dir.glob('*.json')):
        image_path = image_dir / f'{label_path.stem}.png'
        if image_path.exists():
            records.append((image_path, label_path))
    if not records:
        raise FileNotFoundError(f'No PNG/JSON pairs found in {image_dir} and {label_dir}')
    return records


def split_records_by_slide(records, val_fraction=0.1, seed=242):
    """Split records without leaking patches from a slide across train/val."""
    groups = {}
    for record in records:
        groups.setdefault(slide_id_from_stem(record[0].stem), []).append(record)
    group_names = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(group_names)
    val_group_count = max(1, round(len(group_names) * val_fraction))
    val_groups = set(group_names[:val_group_count])
    train = [record for group in group_names if group not in val_groups for record in groups[group]]
    val = [record for group in group_names if group in val_groups for record in groups[group]]
    return train, val


class IHCHierarchicalDataset(Dataset):
    """Read existing five-class JSON labels for hierarchical training.

    ``was_nonT`` is authoritative: any such annotation is mapped to class 4.
    Images are loaded lazily, avoiding the multi-gigabyte in-memory copy used by
    the original notebook.
    """

    def __init__(self, records, input_size=512, augment=False):
        self.records = list(records)
        self.input_size = int(input_size)
        self.augment = augment

    def __len__(self):
        return len(self.records)

    @staticmethod
    def load_labels(label_path):
        with open(label_path, encoding='utf-8') as file:
            annotations = json.load(file)
        labels = []
        for annotation in annotations:
            class_id = 4 if annotation.get('was_nonT', False) else int(annotation['class_id'])
            if not 0 <= class_id <= 4:
                raise ValueError(f'Invalid class_id={class_id} in {label_path}')
            labels.append([
                class_id,
                float(annotation['cx']), float(annotation['cy']),
                float(annotation['w']), float(annotation['h']),
            ])
        return np.asarray(labels, dtype=np.float32).reshape(-1, 5)

    @staticmethod
    def _color_augmentation(image):
        if random.random() < 0.5:
            image = np.clip(image.astype(np.float32) * random.uniform(0.85, 1.15), 0, 255)
        if random.random() < 0.5:
            hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + random.uniform(-8, 8)) % 180
            hsv[..., 1] *= random.uniform(0.8, 1.2)
            hsv[..., 2] *= random.uniform(0.85, 1.15)
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        if random.random() < 0.2:
            image = cv2.GaussianBlur(image.astype(np.uint8), (3, 3), 0)
        return image.astype(np.uint8)

    def __getitem__(self, index):
        image_path, label_path = self.records[index]
        with Image.open(image_path) as pil_image:
            image = np.asarray(pil_image.convert('RGB'))
        labels = self.load_labels(label_path)

        if image.shape[:2] != (self.input_size, self.input_size):
            image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)

        if self.augment and len(labels):
            if random.random() < 0.5:
                image = np.fliplr(image)
                labels[:, 1] = 1.0 - labels[:, 1]
            if random.random() < 0.5:
                image = np.flipud(image)
                labels[:, 2] = 1.0 - labels[:, 2]
            if random.random() < 0.3:
                rotations = random.randint(1, 3)
                image = np.rot90(image, rotations)
                for _ in range(rotations):
                    old_x, old_y = labels[:, 1].copy(), labels[:, 2].copy()
                    old_w, old_h = labels[:, 3].copy(), labels[:, 4].copy()
                    labels[:, 1], labels[:, 2] = old_y, 1.0 - old_x
                    labels[:, 3], labels[:, 4] = old_h, old_w
            image = self._color_augmentation(image)

        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        classes = torch.from_numpy(labels[:, 0].copy())
        boxes = torch.from_numpy(labels[:, 1:5].copy())
        return torch.from_numpy(image), classes, boxes, torch.zeros(len(labels))

    @staticmethod
    def collate_fn(batch):
        images, classes, boxes, indices = zip(*batch)
        classes = torch.cat(classes)
        boxes = torch.cat(boxes)
        indexed = []
        for image_index, value in enumerate(indices):
            indexed.append(value + image_index)
        return torch.stack(images), {
            'cls': classes,
            'box': boxes,
            'idx': torch.cat(indexed),
        }
