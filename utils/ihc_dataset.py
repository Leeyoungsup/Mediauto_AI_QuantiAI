"""Dataset helpers for the MediautoAI PD-L1 stomach cell data."""

from __future__ import annotations

import json
import random
import re
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


PDL1_CLASS_NAMES = {
    0: "Negative Epithelial",
    1: "Negative Lymphocyte",
    2: "Negative Macrophage",
    3: "Positive Epithelial",
    4: "Positive Lymphocyte",
    5: "Positive Macrophage",
    6: "other",
}
PDL1_CLASS_TO_ID = {name.casefold(): class_id for class_id, name in PDL1_CLASS_NAMES.items()}
PDL1_CLASS_COLORS = {
    0: "#1f77b4",
    1: "#17becf",
    2: "#2ca02c",
    3: "#d62728",
    4: "#ff7f0e",
    5: "#9467bd",
    6: "#7f7f7f",
}
IMAGE_SUFFIXES = (".jpeg", ".jpg", ".png", ".tif", ".tiff")
MAX_CELL_BOX_SIZE_PX = 100.0


def slide_id_from_stem(stem: str) -> str:
    """Return a slide-level group id from legacy ``<slide>_<x>_<y>`` names."""
    match = re.match(r"^(.*\.(?:svs|tif|tiff|ndpi))_\d+_\d+$", stem, flags=re.IGNORECASE)
    return match.group(1) if match else stem.rsplit("_", 2)[0]


def slide_id_from_record(record) -> str:
    """Return the slide id for either nested MediautoAI or legacy records."""
    image_path = Path(record[0])
    if image_path.name.startswith("patch_"):
        return image_path.parent.name
    return slide_id_from_stem(image_path.stem)


def _matching_image(label_path: Path) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        image_path = label_path.with_suffix(suffix)
        if image_path.exists():
            return image_path
    return None


def discover_records(image_dir, label_dir=None):
    """Find image/JSON pairs in the nested PD-L1 or legacy two-folder layout.

    Preferred usage is ``discover_records(data_root)`` for records stored as
    ``<slide>/patch_N.jpeg`` and ``<slide>/patch_N.json``.  The old
    ``discover_records(image_dir, label_dir)`` form remains supported.
    """
    image_dir = Path(image_dir)
    records = []

    if label_dir is None:
        for label_path in sorted(image_dir.glob("*/*.json")):
            image_path = _matching_image(label_path)
            if image_path is not None:
                records.append((image_path, label_path))
    else:
        label_dir = Path(label_dir)
        for label_path in sorted(label_dir.glob("*.json")):
            image_path = _matching_image(image_dir / label_path.name)
            if image_path is not None:
                records.append((image_path, label_path))

    if not records:
        location = image_dir if label_dir is None else f"{image_dir} and {label_dir}"
        raise FileNotFoundError(f"No image/JSON pairs found in {location}")
    return records


def split_records_by_slide(records, val_fraction=0.1, seed=242):
    """Split records without leaking patches from a slide across train/val."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    groups = {}
    for record in records:
        groups.setdefault(slide_id_from_record(record), []).append(record)
    group_names = sorted(groups)
    if len(group_names) < 2:
        raise ValueError("At least two slides are required for a train/validation split")
    rng = random.Random(seed)
    rng.shuffle(group_names)
    val_group_count = min(len(group_names) - 1, max(1, round(len(group_names) * val_fraction)))
    val_groups = set(group_names[:val_group_count])
    train = [record for group in group_names if group not in val_groups for record in groups[group]]
    val = [record for group in group_names if group in val_groups for record in groups[group]]
    return train, val


def count_record_classes(records, num_classes=len(PDL1_CLASS_NAMES)):
    """Count supported cell labels in a collection of image/JSON records."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, label_path in records:
        labels = IHCHierarchicalDataset.load_labels(label_path)
        if len(labels):
            counts += np.bincount(
                labels[:, 0].astype(np.int64), minlength=num_classes
            )[:num_classes]
    return counts


def split_records_by_slide_balanced(records, val_slide_count=3):
    """Select validation slides that cover every class as evenly as possible.

    Every slide combination of ``val_slide_count`` is evaluated using the
    fraction of each class assigned to validation. The weakest-covered class
    is maximized first, followed by geometric-mean and mean class coverage.
    Both train and validation must contain every supported class.

    This is a label-stratified development split, not an estimate of natural
    cohort prevalence. It is intended for small cohorts where a random slide
    split can contain too few minority cells to diagnose model learning.
    """
    groups = {}
    for record in records:
        groups.setdefault(slide_id_from_record(record), []).append(record)
    slide_names = sorted(groups)
    if not 0 < val_slide_count < len(slide_names):
        raise ValueError(
            "val_slide_count must be positive and smaller than the slide count"
        )

    slide_counts = {
        slide_name: count_record_classes(groups[slide_name])
        for slide_name in slide_names
    }
    total_counts = sum(
        slide_counts.values(), np.zeros(len(PDL1_CLASS_NAMES), dtype=np.int64)
    )
    if np.any(total_counts == 0):
        missing = np.flatnonzero(total_counts == 0).tolist()
        raise ValueError(f"Dataset has no labels for classes: {missing}")

    best_score = None
    best_slides = None
    for candidate in combinations(slide_names, val_slide_count):
        val_counts = sum(
            (slide_counts[name] for name in candidate),
            np.zeros_like(total_counts),
        )
        train_counts = total_counts - val_counts
        if np.any(val_counts == 0) or np.any(train_counts == 0):
            continue
        coverage = val_counts / total_counts
        score = (
            float(coverage.min()),
            float(np.exp(np.log(coverage).mean())),
            float(coverage.mean()),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_slides = candidate

    if best_slides is None:
        raise ValueError(
            "No slide combination leaves every class in both train and validation"
        )

    val_slide_names = set(best_slides)
    train = [
        record for name in slide_names if name not in val_slide_names
        for record in groups[name]
    ]
    val = [
        record for name in slide_names if name in val_slide_names
        for record in groups[name]
    ]
    return train, val


class IHCHierarchicalDataset(Dataset):
    """Load resized PD-L1 patches and return a 512-pixel crop for detection.

    Images are resized to ``resize_size`` once and, by default, kept as uint8
    RGB arrays in RAM. Training uses a random crop. With ``four_crop=True``,
    each 1024 image becomes four non-overlapping 512 validation samples. Only
    the seven names in :data:`PDL1_CLASS_NAMES` are kept. Boxes are clipped to
    the crop and objects whose centers fall outside it are discarded.
    """

    class_names = PDL1_CLASS_NAMES
    class_colors = PDL1_CLASS_COLORS

    def __init__(self, records, input_size=512, resize_size=1024, augment=False,
                 random_crop=None, cache_images=True, four_crop=False,
                 class_aware_crop_probability=0.0, class_sampling_weights=None):
        self.records = list(records)
        self.input_size = int(input_size)
        self.resize_size = int(resize_size)
        self.augment = bool(augment)
        self.random_crop = self.augment if random_crop is None else bool(random_crop)
        self.cache_images = bool(cache_images)
        self.four_crop = bool(four_crop)
        self.class_aware_crop_probability = float(class_aware_crop_probability)
        if class_sampling_weights is None:
            class_sampling_weights = np.ones(len(PDL1_CLASS_NAMES), dtype=np.float32)
        self.class_sampling_weights = np.asarray(
            class_sampling_weights, dtype=np.float32
        )

        if self.input_size <= 0 or self.resize_size <= 0:
            raise ValueError("input_size and resize_size must be positive")
        if self.input_size > self.resize_size:
            raise ValueError("input_size cannot be larger than resize_size")
        if self.four_crop and self.resize_size != 2 * self.input_size:
            raise ValueError("four_crop requires resize_size == 2 * input_size")
        if self.four_crop and self.random_crop:
            raise ValueError("four_crop and random_crop cannot be enabled together")
        if not 0.0 <= self.class_aware_crop_probability <= 1.0:
            raise ValueError("class_aware_crop_probability must be between 0 and 1")
        if self.class_sampling_weights.shape != (len(PDL1_CLASS_NAMES),):
            raise ValueError("class_sampling_weights must contain one value per class")
        if np.any(self.class_sampling_weights < 0):
            raise ValueError("class_sampling_weights cannot be negative")

        self._image_cache = [None] * len(self.records)
        self._label_cache = [None] * len(self.records)
        if self.cache_images:
            for index in range(len(self.records)):
                image, labels = self._load_resized_pair(index)
                self._image_cache[index] = image
                self._label_cache[index] = labels

    def __len__(self):
        return len(self.records) * (4 if self.four_crop else 1)

    @property
    def cache_memory_bytes(self):
        """Bytes occupied by cached image arrays (labels excluded)."""
        return sum(image.nbytes for image in self._image_cache if image is not None)

    @staticmethod
    def _mediauto_box(cell, image_width, image_height):
        coordinates = cell.get("local_coordinates")
        if coordinates:
            points = np.asarray(coordinates, dtype=np.float32).reshape(-1, 2)
            x0, y0 = points.min(axis=0)
            x1, y1 = points.max(axis=0)
        elif "local_x" in cell and "local_y" in cell:
            bbox = cell.get("bbox", {})
            width = float(bbox.get("width", 0.0))
            height = float(bbox.get("height", 0.0))
            cx, cy = float(cell["local_x"]), float(cell["local_y"])
            x0, y0, x1, y1 = cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2
        else:
            return None

        # Cell boxes larger than 100 px in either dimension are annotation
        # errors. Apply the limit in the source-image coordinate system before
        # clipping or resizing so every downstream consumer sees the same set.
        if x1 - x0 > MAX_CELL_BOX_SIZE_PX or y1 - y0 > MAX_CELL_BOX_SIZE_PX:
            return None

        x0, x1 = np.clip([x0, x1], 0, image_width)
        y0, y1 = np.clip([y0, y1], 0, image_height)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    @classmethod
    def load_labels(cls, label_path, image_size=None):
        """Load labels as normalized ``[class, cx, cy, width, height]`` rows."""
        label_path = Path(label_path)
        with label_path.open(encoding="utf-8") as file:
            annotations = json.load(file)

        # Backward-compatible reader for the original list-based JSON format.
        if isinstance(annotations, list):
            labels = []
            for annotation in annotations:
                class_id = 4 if annotation.get("was_nonT", False) else int(annotation["class_id"])
                labels.append([
                    class_id,
                    float(annotation["cx"]), float(annotation["cy"]),
                    float(annotation["w"]), float(annotation["h"]),
                ])
            return np.asarray(labels, dtype=np.float32).reshape(-1, 5)

        if image_size is None:
            image_path = _matching_image(label_path)
            if image_path is None:
                raise FileNotFoundError(f"No image found for {label_path}")
            with Image.open(image_path) as image:
                image_size = image.size
        image_width, image_height = map(float, image_size)

        labels = []
        for cell in annotations.get("cells", []):
            # class_name is authoritative. Unknown labels (for example Tumor)
            # are intentionally ignored even if their class_id looks usable.
            class_name = str(cell.get("class_name", "")).strip().casefold()
            class_id = PDL1_CLASS_TO_ID.get(class_name)
            if class_id is None:
                continue
            box = cls._mediauto_box(cell, image_width, image_height)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            labels.append([
                class_id,
                (x0 + x1) / (2 * image_width),
                (y0 + y1) / (2 * image_height),
                (x1 - x0) / image_width,
                (y1 - y0) / image_height,
            ])
        return np.asarray(labels, dtype=np.float32).reshape(-1, 5)

    def _load_resized_pair(self, index):
        image_path, label_path = self.records[index]
        with Image.open(image_path) as pil_image:
            image_width, image_height = pil_image.size
            image = np.asarray(pil_image.convert("RGB"))
        labels = self.load_labels(label_path, image_size=(image_width, image_height))
        if image.shape[:2] != (self.resize_size, self.resize_size):
            interpolation = cv2.INTER_AREA if max(image.shape[:2]) > self.resize_size else cv2.INTER_LINEAR
            image = cv2.resize(image, (self.resize_size, self.resize_size), interpolation=interpolation)
        return np.ascontiguousarray(image), labels

    def _get_resized_pair(self, index):
        if self.cache_images:
            return self._image_cache[index], self._label_cache[index].copy()
        return self._load_resized_pair(index)

    def _class_aware_offsets(self, labels, max_offset):
        """Choose a crop centered around one sampled class when possible."""
        if not len(labels) or self.class_aware_crop_probability <= 0:
            return None
        if random.random() >= self.class_aware_crop_probability:
            return None

        label_classes = labels[:, 0].astype(np.int64)
        present_classes = np.unique(label_classes)
        present_weights = self.class_sampling_weights[present_classes]
        if present_weights.sum() <= 0:
            return None

        target_class = random.choices(
            present_classes.tolist(), weights=present_weights.tolist(), k=1
        )[0]
        target_indices = np.flatnonzero(label_classes == target_class)
        target_index = random.choice(target_indices.tolist())
        center_x = float(labels[target_index, 1] * self.resize_size)
        center_y = float(labels[target_index, 2] * self.resize_size)

        def central_offset(center):
            # Keep the sampled cell in the middle 50% of the crop where the
            # image boundary allows it. This preserves more cell morphology
            # than placing the selected center directly on a crop edge.
            low = max(0, int(np.ceil(center - 0.75 * self.input_size)))
            high = min(max_offset, int(np.floor(center - 0.25 * self.input_size)))
            if low <= high:
                return random.randint(low, high)

            # Boundary fallback: at least guarantee center inclusion.
            low = max(0, int(np.floor(center - self.input_size)) + 1)
            high = min(max_offset, int(np.floor(center)))
            return random.randint(low, high)

        return central_offset(center_x), central_offset(center_y)

    def _crop(self, image, labels, tile_index=None):
        max_offset = self.resize_size - self.input_size
        if tile_index is not None:
            tile_offsets = (
                (0, 0),
                (self.input_size, 0),
                (0, self.input_size),
                (self.input_size, self.input_size),
            )
            left, top = tile_offsets[tile_index]
        elif self.random_crop and max_offset:
            class_aware_offsets = self._class_aware_offsets(labels, max_offset)
            if class_aware_offsets is None:
                left = random.randint(0, max_offset)
                top = random.randint(0, max_offset)
            else:
                left, top = class_aware_offsets
        else:
            left = top = max_offset // 2
        right, bottom = left + self.input_size, top + self.input_size
        cropped = image[top:bottom, left:right]

        if not len(labels):
            return cropped, labels

        boxes = labels[:, 1:5].copy()
        cx = boxes[:, 0] * self.resize_size
        cy = boxes[:, 1] * self.resize_size
        width = boxes[:, 2] * self.resize_size
        height = boxes[:, 3] * self.resize_size
        keep = (cx >= left) & (cx < right) & (cy >= top) & (cy < bottom)
        labels = labels[keep].copy()
        if not len(labels):
            return cropped, labels

        cx, cy, width, height = cx[keep], cy[keep], width[keep], height[keep]
        x0 = np.clip(cx - width / 2, left, right) - left
        y0 = np.clip(cy - height / 2, top, bottom) - top
        x1 = np.clip(cx + width / 2, left, right) - left
        y1 = np.clip(cy + height / 2, top, bottom) - top
        labels[:, 1] = (x0 + x1) / (2 * self.input_size)
        labels[:, 2] = (y0 + y1) / (2 * self.input_size)
        labels[:, 3] = (x1 - x0) / self.input_size
        labels[:, 4] = (y1 - y0) / self.input_size
        return cropped, labels

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
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.four_crop:
            record_index, tile_index = divmod(index, 4)
        else:
            record_index, tile_index = index, None
        image, labels = self._get_resized_pair(record_index)
        image, labels = self._crop(image, labels, tile_index=tile_index)

        if self.augment:
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
        indexed = [value + image_index for image_index, value in enumerate(indices)]
        return torch.stack(images), {
            "cls": torch.cat(classes),
            "box": torch.cat(boxes),
            "idx": torch.cat(indexed),
        }
