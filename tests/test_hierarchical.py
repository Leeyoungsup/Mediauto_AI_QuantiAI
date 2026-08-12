import torch

from nets import nn
from utils import util
from utils.hierarchical import (
    HierarchicalComputeLoss,
    apply_tumor_gate,
    load_flat_checkpoint_into_hierarchical,
)


def test_hierarchical_probabilities_sum_to_cell_probability():
    objectness = torch.tensor([[[0.2, -0.4]]])
    tumor = torch.tensor([[[0.7, -1.0]]])
    grade = torch.randn(1, 4, 2)
    probabilities = nn.HierarchicalHead.probabilities(objectness, tumor, grade)
    assert torch.allclose(probabilities.sum(dim=1), objectness.sigmoid().squeeze(1))


def test_forward_loss_backward_and_eval_shape():
    model = nn.yolo_v11_n_hierarchical()
    images = torch.rand(2, 3, 128, 128)
    targets = {
        'idx': torch.tensor([0.0, 0.0, 1.0]),
        'cls': torch.tensor([4.0, 2.0, 0.0]),
        'box': torch.tensor([
            [0.25, 0.25, 0.10, 0.10],
            [0.65, 0.65, 0.10, 0.10],
            [0.50, 0.50, 0.15, 0.15],
        ]),
    }
    model.train()
    criterion = HierarchicalComputeLoss(model, {'top_k': 5})
    losses = criterion(model(images), targets)
    assert set(losses) == {'box', 'objectness', 'tumor', 'grade', 'dfl', 'total'}
    assert torch.isfinite(losses['total'])
    losses['total'].backward()

    model.eval()
    with torch.no_grad():
        predictions = model(images)
    assert predictions.shape[1] == 9  # xywh + four grades + other


def test_grade_loss_is_masked_for_non_tumor_cells():
    model = nn.yolo_v11_n_hierarchical()
    model.train()
    images = torch.rand(1, 3, 512, 512)
    targets = {
        'idx': torch.tensor([0.0]),
        'cls': torch.tensor([4.0]),
        'box': torch.tensor([[0.5, 0.5, 0.08, 0.08]]),
    }
    losses = HierarchicalComputeLoss(model, {'top_k': 5})(model(images), targets)
    assert losses['grade'].item() == 0.0


def test_nms_defaults_to_one_mutually_exclusive_class():
    # One anchor with two class scores above threshold must yield one detection.
    outputs = torch.tensor([[[10.0], [10.0], [4.0], [4.0], [0.9], [0.8]]])
    result = util.non_max_suppression(outputs, confidence_threshold=0.25)
    assert result[0].shape[0] == 1
    assert result[0][0, 5].item() == 0


def test_explicit_tumor_gate_routes_low_probability_to_other():
    components = {
        'predictions': torch.zeros(1, 9, 2),
        'objectness': torch.tensor([[[0.9, 0.9]]]),
        'tumor_probability': torch.tensor([[[0.4, 0.8]]]),
        'grade_probability': torch.tensor([[[0.7, 0.1], [0.1, 0.7],
                                             [0.1, 0.1], [0.1, 0.1]]]),
    }
    predictions = apply_tumor_gate(components, tumor_threshold=0.6)
    assert predictions[0, 8, 0] > 0
    assert predictions[0, 4:8, 0].sum() == 0
    assert predictions[0, 8, 1] == 0
    assert predictions[0, 4:8, 1].sum() > 0


def test_flat_checkpoint_warm_start():
    flat_model = nn.yolo_v11_n(num_classes=5)
    hierarchical_model = nn.yolo_v11_n_hierarchical()
    report = load_flat_checkpoint_into_hierarchical(
        hierarchical_model, {'model_state_dict': flat_model.state_dict()}
    )
    assert report['loaded_fraction'] == 1.0
    assert torch.allclose(
        hierarchical_model.head.grade[0].weight,
        flat_model.head.cls[0][4].weight[:4],
    )
