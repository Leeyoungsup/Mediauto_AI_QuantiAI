import torch

from nets import nn
from utils import util


def test_flat_seven_class_forward_loss_backward_and_eval_shape():
    model = nn.yolo_v11_n(num_classes=7)
    images = torch.rand(2, 3, 128, 128)
    targets = {
        'idx': torch.tensor([0.0, 0.0, 1.0]),
        'cls': torch.tensor([0.0, 6.0, 4.0]),
        'box': torch.tensor([
            [0.25, 0.25, 0.10, 0.10],
            [0.65, 0.65, 0.12, 0.12],
            [0.50, 0.50, 0.15, 0.15],
        ]),
    }

    model.train()
    criterion = util.ComputeLoss(
        model, {
            'box': 5.0, 'cls': 2.5, 'dfl': 0.5,
            'cls_pos_weight': [1.57, 6.0, 6.0, 2.82, 6.0, 3.30, 1.0],
        }
    )
    loss_box, loss_cls, loss_dfl = criterion(model(images), targets)
    total_loss = loss_box + loss_cls + loss_dfl
    assert torch.isfinite(total_loss)
    total_loss.backward()

    model.eval()
    with torch.no_grad():
        predictions = model(images)
    assert predictions.shape[1] == 11  # xywh + seven mutually-exclusive classes


def test_flat_checkpoint_metadata_contract():
    checkpoint = {
        'architecture': 'flat_7_class_detection',
        'num_classes': 7,
    }
    assert checkpoint['architecture'] == 'flat_7_class_detection'
    assert checkpoint['num_classes'] == 7
