import json

import numpy as np
from PIL import Image

from utils.ihc_dataset import (
    IHCHierarchicalDataset,
    PDL1_CLASS_NAMES,
    discover_records,
    count_record_classes,
    slide_id_from_record,
    split_records_by_slide,
    split_records_by_slide_balanced,
)


def _write_pair(root, slide_name, patch_name="patch_1"):
    slide_dir = root / slide_name
    slide_dir.mkdir(parents=True)
    image_path = slide_dir / f"{patch_name}.jpeg"
    label_path = slide_dir / f"{patch_name}.json"
    Image.fromarray(np.full((200, 200, 3), 127, dtype=np.uint8)).save(image_path)
    cells = []
    for class_id, class_name in PDL1_CLASS_NAMES.items():
        cells.append({
            "class_id": "other" if class_id == 6 else str(class_id),
            "class_name": "Other" if class_name == "other" else class_name,
            "local_coordinates": [[90, 90], [110, 90], [110, 110], [90, 110]],
        })
    cells.append({
        "class_id": "class_8",
        "class_name": "Tumor",
        "local_coordinates": [[90, 90], [110, 90], [110, 110], [90, 110]],
    })
    label_path.write_text(json.dumps({"cells": cells}), encoding="utf-8")
    return image_path, label_path


def test_nested_discovery_class_filter_cache_and_center_crop(tmp_path):
    _write_pair(tmp_path, "slide_a")
    records = discover_records(tmp_path)
    dataset = IHCHierarchicalDataset(
        records, resize_size=1024, input_size=512,
        augment=False, cache_images=True,
    )

    image, classes, boxes, indices = dataset[0]
    assert image.shape == (3, 512, 512)
    assert classes.tolist() == list(range(7))
    assert boxes.shape == (7, 4)
    assert np.allclose(boxes[:, :2].numpy(), 0.5)
    assert indices.shape == (7,)
    assert dataset._image_cache[0].shape == (1024, 1024, 3)
    assert dataset.cache_memory_bytes == 1024 * 1024 * 3


def test_source_boxes_larger_than_100_pixels_are_filtered(tmp_path):
    _, label_path = _write_pair(tmp_path, "slide_a")
    data = json.loads(label_path.read_text(encoding="utf-8"))
    data["cells"] = [
        {
            "class_name": "Negative Epithelial",
            "local_coordinates": [[10, 10], [110, 10], [110, 110], [10, 110]],
        },
        {
            "class_name": "Positive Epithelial",
            "local_coordinates": [[10, 10], [111, 10], [111, 30], [10, 30]],
        },
        {
            "class_name": "Positive Lymphocyte",
            "local_coordinates": [[10, 10], [30, 10], [30, 111], [10, 111]],
        },
    ]
    label_path.write_text(json.dumps(data), encoding="utf-8")

    labels = IHCHierarchicalDataset.load_labels(label_path)

    # Exactly 100x100 is valid; either dimension above 100 is rejected.
    assert labels[:, 0].tolist() == [0.0]


def test_random_crop_updates_and_filters_boxes(tmp_path, monkeypatch):
    _, label_path = _write_pair(tmp_path, "slide_a")
    data = json.loads(label_path.read_text(encoding="utf-8"))
    data["cells"] = [
        {
            "class_id": "0",
            "class_name": "Negative Epithelial",
            "local_coordinates": [[20, 20], [40, 20], [40, 40], [20, 40]],
        },
        {
            "class_id": "3",
            "class_name": "Positive Epithelial",
            "local_coordinates": [[160, 160], [180, 160], [180, 180], [160, 180]],
        },
    ]
    label_path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr("utils.ihc_dataset.random.randint", lambda _low, _high: 0)

    dataset = IHCHierarchicalDataset(
        discover_records(tmp_path), resize_size=1024, input_size=512,
        random_crop=True, augment=False, cache_images=False,
    )
    _, classes, boxes, _ = dataset[0]
    assert classes.tolist() == [0.0]
    assert np.allclose(boxes[0, :2].numpy(), [0.3, 0.3], atol=1e-6)


def test_slide_split_uses_parent_folder_for_patch_names(tmp_path):
    records = [
        _write_pair(tmp_path, "slide_a", "patch_1"),
        _write_pair(tmp_path, "slide_a", "patch_2"),
        _write_pair(tmp_path, "slide_b", "patch_1"),
    ]
    train, val = split_records_by_slide(records, val_fraction=0.5, seed=1)
    train_slides = {slide_id_from_record(record) for record in train}
    val_slides = {slide_id_from_record(record) for record in val}
    assert train_slides.isdisjoint(val_slides)
    assert train_slides | val_slides == {"slide_a", "slide_b"}


def test_balanced_slide_split_covers_all_classes_without_leakage(tmp_path):
    records = []
    for slide_index in range(4):
        records.append(_write_pair(tmp_path, f"slide_{slide_index}"))
    train, val = split_records_by_slide_balanced(records, val_slide_count=2)
    train_slides = {slide_id_from_record(record) for record in train}
    val_slides = {slide_id_from_record(record) for record in val}
    assert train_slides.isdisjoint(val_slides)
    assert len(val_slides) == 2
    assert np.all(count_record_classes(train) > 0)
    assert np.all(count_record_classes(val) > 0)


def test_four_crop_expands_each_validation_patch_without_label_duplicates(tmp_path):
    _write_pair(tmp_path, "slide_a")
    records = discover_records(tmp_path)
    source_labels = IHCHierarchicalDataset.load_labels(records[0][1])
    dataset = IHCHierarchicalDataset(
        records, resize_size=1024, input_size=512,
        augment=False, random_crop=False, cache_images=True, four_crop=True,
    )

    assert len(dataset) == 4
    tile_counts = []
    for tile_index in range(4):
        image, classes, boxes, _ = dataset[tile_index]
        assert image.shape == (3, 512, 512)
        assert boxes.numel() == 0 or bool(((boxes >= 0) & (boxes <= 1)).all())
        tile_counts.append(len(classes))

    # Centers on the 512 boundary are assigned to the bottom-right tile only.
    assert tile_counts == [0, 0, 0, len(source_labels)]
    assert sum(tile_counts) == len(source_labels)
    assert dataset.cache_memory_bytes == 1024 * 1024 * 3


def test_class_aware_crop_keeps_selected_minority_cell(tmp_path):
    _, label_path = _write_pair(tmp_path, "slide_a")
    data = json.loads(label_path.read_text(encoding="utf-8"))
    data["cells"] = [
        {
            "class_name": "Negative Epithelial",
            "local_coordinates": [[10, 10], [30, 10], [30, 30], [10, 30]],
        },
        {
            "class_name": "Positive Lymphocyte",
            "local_coordinates": [[160, 160], [180, 160], [180, 180], [160, 180]],
        },
    ]
    label_path.write_text(json.dumps(data), encoding="utf-8")

    dataset = IHCHierarchicalDataset(
        discover_records(tmp_path), resize_size=1024, input_size=512,
        random_crop=True, augment=False, cache_images=False,
        class_aware_crop_probability=1.0,
        class_sampling_weights=[0, 0, 0, 0, 1, 0, 0],
    )
    _, classes, boxes, _ = dataset[0]
    assert 4.0 in classes.tolist()
    target_box = boxes[classes == 4.0][0]
    assert 0.25 <= target_box[0] <= 0.75
    assert 0.25 <= target_box[1] <= 0.75
