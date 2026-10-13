import torch

from utils.pdl1_merged_immune import (
    CLASS_NAMES, MergedImmuneDataset, PDL1MergedImmuneHead,
    PDL1MergedImmuneYOLO, PDL1MergedImmuneLoss,
)
from utils.valid import compute_point_label_metrics_single


def test_merge_labels_preserves_polarity_and_boxes():
    image = torch.zeros(3, 32, 32)
    boxes = torch.rand(7, 4)
    dataset = MergedImmuneDataset([(image, torch.arange(7).float(), boxes, torch.zeros(7))])
    actual_image, labels, actual_boxes, _ = dataset[0]
    assert labels.tolist() == [0, 1, 1, 2, 3, 3, 4]
    assert actual_image is image and actual_boxes is boxes


def test_scores_conserve_objectness_and_merge_original_subtypes():
    from nets.nn import PDL1HierarchicalHead
    obj, root = torch.randn(2, 1, 10), torch.randn(2, 3, 10)
    epi, subtype, polarity = [torch.randn(2, 1, 10) for _ in range(3)]
    old = PDL1HierarchicalHead.probabilities(obj, root, epi, subtype, polarity)
    merged = PDL1MergedImmuneHead.probabilities(obj, root, epi, polarity)
    expected = torch.stack((old[:, 0], old[:, 1] + old[:, 2], old[:, 3], old[:, 4] + old[:, 5], old[:, 6]), dim=1)
    assert torch.allclose(merged, expected, atol=1e-6)
    assert torch.allclose(merged.sum(1), obj.sigmoid().squeeze(1), atol=1e-6)


def test_training_branch_masks_empty_targets_and_validation():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = PDL1MergedImmuneYOLO([3, 16, 32, 64, 128, 256], [1]*6, [False, True])
    assert not hasattr(model.head, 'immune_lymphocyte')
    criterion = PDL1MergedImmuneLoss(model, {'top_k': 5})
    images = torch.rand(1, 3, 128, 128)
    for cls in range(5):
        targets = {'idx': torch.tensor([0.]), 'cls': torch.tensor([float(cls)]),
                   'box': torch.tensor([[.5, .5, .25, .25]])}
        model.train()
        model.zero_grad(set_to_none=True)
        losses = criterion(model(images), targets)
        assert 'immune_type' not in losses
        assert torch.isfinite(losses['total'])
        if cls in (0, 2):
            assert losses['epithelial'] > 0
            assert losses['immune_polarity'] == 0
        elif cls in (1, 3):
            assert losses['immune_polarity'] > 0
            assert losses['epithelial'] == 0
        else:
            assert losses['epithelial'] == losses['immune_polarity'] == 0
        losses['total'].backward()
    empty = {'idx': torch.empty(0), 'cls': torch.empty(0), 'box': torch.empty(0, 4)}
    losses = criterion(model(images), empty)
    assert torch.isfinite(losses['total'])
    losses['total'].backward()
    model.eval()
    with torch.no_grad():
        assert model(images).shape[1] == 9  # box + five scores
    metrics = compute_point_label_metrics_single(
        model, [(images * 255, targets)], torch.device('cpu'), {'names': CLASS_NAMES},
        class_agnostic_nms=True, confidence_threshold=0.1,
    )
    assert set(metrics['class_stats']) == set(CLASS_NAMES.values())
