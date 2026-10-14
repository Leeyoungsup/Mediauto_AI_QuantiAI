"""Test the training notebook's Dataset-level empty-crop handling."""
import ast
import json
from pathlib import Path
import torch


def dataset_class():
    path = Path(__file__).resolve().parents[1] / 'Precise_PDL1_ST_flat_5class_train.ipynb'
    notebook = json.loads(path.read_text())
    source = ''.join(notebook['cells'][2]['source'])
    node = ast.parse(source).body[0]
    context = {'torch': torch, 'CLASS_NAMES': {}, 'CLASS_COLORS': {},
               'SOURCE_TO_FLAT': (0, 1, 1, 2, 3, 3, 4)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), context)
    return context['Flat5ClassDataset']


class FakeCrops:
    four_crop = False
    random_crop = True
    cache_memory_bytes = 0
    def __init__(self, always_empty=False):
        self.calls = []
        self.always_empty = always_empty
    def __len__(self):
        return 2
    def _get_resized_pair(self, index):
        return None, ([] if index == 0 else [[3, .5, .5, .1, .1]])
    def __getitem__(self, index):
        self.calls.append(index)
        # First crop empty, next crop annotated. Source 0 has no labels.
        count = 0 if index == 0 or self.always_empty or len(self.calls) == 1 else 1
        return torch.zeros(3, 8, 8), torch.full((count,), 3.), torch.ones(count, 4), torch.zeros(count)


def test_training_excludes_empty_sources_and_resamples_empty_crops():
    base = FakeCrops()
    dataset = dataset_class()(base, require_labels=True)
    assert len(dataset) == 1 and dataset.excluded_empty_records == 1
    image, labels, boxes, indices = dataset[0]
    assert base.calls == [1, 1]
    assert labels.tolist() == [2]  # Positive Epithelial remapping
    assert boxes.shape == (1, 4) and indices.tolist() == [0]


def test_validation_keeps_empty_crops_without_retry():
    base = FakeCrops()
    dataset = dataset_class()(base)
    assert len(dataset) == 2
    assert dataset[0][1].numel() == 0
    assert base.calls == [0]


def test_retry_limit_and_all_empty_source_error():
    base = FakeCrops(always_empty=True)
    dataset = dataset_class()(base, require_labels=True, max_crop_attempts=3)
    try:
        dataset[0]
    except RuntimeError:
        assert base.calls == [1, 1, 1]
    else:
        raise AssertionError('Expected bounded retry failure')
    base._get_resized_pair = lambda index: (None, [])
    try:
        dataset_class()(base, require_labels=True)
    except ValueError:
        pass
    else:
        raise AssertionError('All-empty training sources must fail early')
