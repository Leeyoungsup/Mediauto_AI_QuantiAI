import torch

from nets import nn
from utils.hierarchical import (
    PDL1HierarchicalComputeLoss,
    load_flat_checkpoint_into_pdl1_hierarchical,
)
from utils.valid import compute_point_label_metrics_single


def _params():
    return {
        'box': 5.0,
        'dfl': 0.5,
        'objectness': 1.0,
        'root': 1.0,
        'epithelial': 1.0,
        'immune_type': 1.0,
        'immune_polarity': 1.0,
        'root_class_weights': [1.5, 2.8, 1.0],
        'epithelial_positive_weight': 2.0,
        'immune_lymphocyte_weight': 2.0,
        'immune_negative_weight': 2.0,
        'top_k': 5,
    }


def test_hierarchical_probabilities_sum_to_objectness():
    objectness = torch.randn(2, 1, 20)
    root = torch.randn(2, 3, 20)
    epithelial = torch.randn(2, 1, 20)
    immune_type = torch.randn(2, 1, 20)
    immune_polarity = torch.randn(2, 1, 20)
    probabilities = nn.PDL1HierarchicalHead.probabilities(
        objectness, root, epithelial, immune_type, immune_polarity
    )
    assert probabilities.shape == (2, 7, 20)
    assert torch.allclose(
        probabilities.sum(dim=1), objectness.sigmoid().squeeze(1), atol=1e-6
    )


def test_pdl1_hierarchy_forward_loss_backward_and_eval_shape():
    model = nn.yolo_v11_n_pdl1_hierarchical()
    images = torch.rand(2, 3, 128, 128)
    targets = {
        'idx': torch.tensor([0.0, 0.0, 1.0, 1.0]),
        'cls': torch.tensor([0.0, 4.0, 2.0, 6.0]),
        'box': torch.tensor([
            [0.25, 0.25, 0.12, 0.12],
            [0.65, 0.65, 0.10, 0.10],
            [0.35, 0.35, 0.10, 0.10],
            [0.70, 0.70, 0.14, 0.14],
        ]),
    }
    model.train()
    losses = PDL1HierarchicalComputeLoss(model, _params())(
        model(images), targets
    )
    assert set(losses) == {
        'box', 'objectness', 'root', 'epithelial', 'immune_type',
        'immune_polarity', 'dfl', 'total',
    }
    assert torch.isfinite(losses['total'])
    losses['total'].backward()

    model.eval()
    with torch.no_grad():
        predictions = model(images)
        components = model(images, return_components=True)
    assert predictions.shape[1] == 11  # xywh + seven classes
    assert components['root_probability'].shape[1] == 3
    assert torch.allclose(
        components['root_probability'].sum(dim=1),
        torch.ones_like(components['root_probability'][:, 0]), atol=1e-6,
    )


def test_branch_losses_are_masked_by_root_group():
    model = nn.yolo_v11_n_pdl1_hierarchical().train()
    images = torch.rand(1, 3, 128, 128)
    criterion = PDL1HierarchicalComputeLoss(model, _params())

    epithelial_targets = {
        'idx': torch.tensor([0.0]),
        'cls': torch.tensor([3.0]),
        'box': torch.tensor([[0.5, 0.5, 0.15, 0.15]]),
    }
    epithelial_losses = criterion(model(images), epithelial_targets)
    assert epithelial_losses['epithelial'].item() > 0
    assert epithelial_losses['immune_type'].item() == 0
    assert epithelial_losses['immune_polarity'].item() == 0

    immune_targets = {
        'idx': torch.tensor([0.0]),
        'cls': torch.tensor([1.0]),
        'box': torch.tensor([[0.5, 0.5, 0.15, 0.15]]),
    }
    immune_losses = criterion(model(images), immune_targets)
    assert immune_losses['epithelial'].item() == 0
    assert immune_losses['immune_type'].item() > 0
    assert immune_losses['immune_polarity'].item() > 0


def test_flat_checkpoint_warm_starts_new_hierarchy():
    flat_model = nn.yolo_v11_n(num_classes=7)
    hierarchical_model = nn.yolo_v11_n_pdl1_hierarchical()
    report = load_flat_checkpoint_into_pdl1_hierarchical(
        hierarchical_model, {'model_state_dict': flat_model.state_dict()}
    )
    assert report['loaded_fraction'] > 0.8
    assert torch.allclose(
        hierarchical_model.head.epithelial_positive[0].weight,
        flat_model.head.cls[0][4].weight[3:4]
        - flat_model.head.cls[0][4].weight[0:1],
    )


def test_hierarchy_validation_reports_root_and_branch_metrics():
    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.head = type(
                'HeadMetadata', (),
                {'hierarchy_name': 'pdl1_epithelial_immune_other'},
            )()

        def forward(self, images, return_components=False):
            batch = len(images)
            dtype = images.dtype
            device = images.device
            predictions = torch.zeros(batch, 11, 1, dtype=dtype, device=device)
            predictions[:, :4, 0] = torch.tensor(
                [256, 256, 20, 20], dtype=dtype, device=device
            )
            class_ids = (3, 1, 6)
            for index, class_id in enumerate(class_ids):
                predictions[index, 4 + class_id, 0] = 0.9
            components = {
                'predictions': predictions,
                'root_probability': torch.tensor(
                    [[[0.9], [0.05], [0.05]],
                     [[0.05], [0.9], [0.05]],
                     [[0.05], [0.05], [0.9]]],
                    dtype=dtype, device=device,
                ),
                'epithelial_positive_probability': torch.tensor(
                    [[[0.9]], [[0.1]], [[0.1]]], dtype=dtype, device=device
                ),
                'immune_lymphocyte_probability': torch.tensor(
                    [[[0.1]], [[0.9]], [[0.1]]], dtype=dtype, device=device
                ),
                'immune_negative_probability': torch.tensor(
                    [[[0.1]], [[0.9]], [[0.1]]], dtype=dtype, device=device
                ),
            }
            return components if return_components else predictions

    model = Stub()
    images = torch.zeros(3, 3, 512, 512)
    targets = {
        'idx': torch.tensor([0.0, 1.0, 2.0]),
        'cls': torch.tensor([3.0, 1.0, 6.0]),
        'box': torch.tensor([
            [0.5, 0.5, 20 / 512, 20 / 512],
            [0.5, 0.5, 20 / 512, 20 / 512],
            [0.5, 0.5, 20 / 512, 20 / 512],
        ]),
    }
    metrics = compute_point_label_metrics_single(
        model, [(images, targets)], torch.device('cpu'),
        {'names': {index: str(index) for index in range(7)}},
        class_agnostic_nms=True,
    )
    assert metrics['root_balanced_accuracy'] == 1.0
    assert metrics['epithelial_branch_accuracy'] == 1.0
    assert metrics['immune_type_accuracy'] == 1.0
    assert metrics['immune_polarity_accuracy'] == 1.0
