"""Strict five-class evaluation reader for native-five and legacy-seven labels."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from utils.ihc_dataset import IHCHierarchicalDataset, _matching_image, slide_id_from_record

FLAT5_CLASS_NAMES = {
    0: 'Negative Epithelial', 1: 'Negative Immune',
    2: 'Positive Epithelial', 3: 'Positive Immune', 4: 'other',
}
NAME_TO_FLAT5 = {
    'negative epithelial': 0, 'negative tumor(epithelial)': 0,
    'negative immune': 1, 'negative lymphocyte': 1, 'negative macrophage': 1,
    'positive epithelial': 2, 'positive tumor(epithelial)': 2,
    'positive immune': 3, 'positive lymphocyte': 3, 'positive macrophage': 3,
    'other': 4,
}


class Flat5EvaluationDataset(IHCHierarchicalDataset):
    """Return five-class labels directly; do not remap these through seven IDs."""
    class_names = FLAT5_CLASS_NAMES

    @classmethod
    def load_labels(cls, label_path, image_size=None):
        label_path = Path(label_path)
        annotations = json.loads(label_path.read_text(encoding='utf-8'))
        if not isinstance(annotations, dict) or not isinstance(annotations.get('cells'), list):
            raise ValueError(f'{label_path}: expected a cells list with explicit class_name')
        if image_size is None:
            image_path = _matching_image(label_path)
            if image_path is None:
                raise FileNotFoundError(f'No matching image: {label_path}')
            with Image.open(image_path) as image:
                image_size = image.size
        width, height = image_size
        labels = []
        for cell in annotations['cells']:
            name = str(cell.get('class_name', '')).strip().casefold()
            if name not in NAME_TO_FLAT5:
                raise ValueError(f'{label_path}: unknown class_name={cell.get("class_name")!r}')
            box = cls._mediauto_box(cell, width, height)
            if box is None:
                continue  # Preserve existing invalid/oversized box filtering.
            x0, y0, x1, y1 = box
            labels.append([NAME_TO_FLAT5[name], (x0+x1)/(2*width), (y0+y1)/(2*height),
                           (x1-x0)/width, (y1-y0)/height])
        return np.asarray(labels, dtype=np.float32).reshape(-1, 5)


def select_fixed_test_records(records, data_root, manifest_path):
    """Select exactly the frozen new records; reject missing/old-slide entries."""
    root = Path(data_root).resolve()
    manifest = json.loads(Path(manifest_path).read_text())
    wanted = manifest['image_paths']
    if len(wanted) != manifest['expected_count'] or len(set(wanted)) != len(wanted):
        raise ValueError('Invalid test manifest count or duplicate paths')
    available = {str(Path(image).resolve().relative_to(root)): (image, label) for image, label in records}
    missing = sorted(set(wanted) - set(available))
    if missing:
        raise FileNotFoundError(f'Missing test image/label pairs: {missing}')
    selected = [available[name] for name in wanted]
    old_slides = set(manifest['excluded_previous_slides'])
    if any(slide_id_from_record(record) in old_slides for record in selected):
        raise ValueError('Selected test records overlap previous evaluation slides')
    return selected
