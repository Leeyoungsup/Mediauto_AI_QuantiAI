"""Five-class PD-L1 hierarchy: epithelial/immune polarity, without immune subtype."""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from nets.nn import PDL1HierarchicalHead, PDL1HierarchicalYOLO, DarkNet, DarkFPN
from utils import util
from utils.util import make_anchors
from utils.hierarchical import PDL1HierarchicalComputeLoss

CLASS_NAMES = {
    0: 'Negative Epithelial', 1: 'Negative Immune',
    2: 'Positive Epithelial', 3: 'Positive Immune', 4: 'other',
}
CLASS_COLORS = {0: '#1f77b4', 1: '#2ca02c', 2: '#d62728', 3: '#ff7f0e', 4: '#7f7f7f'}
SOURCE_TO_MERGED = (0, 1, 1, 2, 3, 3, 4)


class MergedImmuneDataset(torch.utils.data.Dataset):
    class_names = CLASS_NAMES
    class_colors = CLASS_COLORS

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, classes, boxes, indices = self.base_dataset[index]
        mapping = classes.new_tensor(SOURCE_TO_MERGED)
        return image, mapping[classes.long()], boxes, indices

    @property
    def cache_memory_bytes(self):
        return self.base_dataset.cache_memory_bytes


class PDL1MergedImmuneHead(PDL1HierarchicalHead):
    nc = 5
    hierarchy_name = 'pdl1_merged_immune'

    def __init__(self, filters=()):
        super().__init__(filters)
        del self.immune_lymphocyte
        self.num_semantic_outputs = 1 + 3 + 1 + 1
        self.no = self.ch * 4 + self.num_semantic_outputs

    @staticmethod
    def probabilities(
            objectness_logits, root_logits, epithelial_positive_logits,
            immune_negative_logits, dim=1):
        """Combine conditional logits into five mutually-exclusive scores."""
        cell = objectness_logits.sigmoid()
        root = root_logits.softmax(dim=dim)
        epithelial, immune, other = root.split(1, dim=dim)
        epithelial_positive = epithelial_positive_logits.sigmoid()
        immune_negative = immune_negative_logits.sigmoid()
        return torch.cat((
            cell * epithelial * (1 - epithelial_positive),
            cell * immune * immune_negative,
            cell * epithelial * epithelial_positive,
            cell * immune * (1 - immune_negative),
            cell * other,
        ), dim=dim)

    def forward(self, features, return_components=False):
        raw = []
        for feature, box, semantic, objectness, root, epithelial, immune_polarity in zip(
                features, self.box, self.semantic, self.objectness, self.root,
                self.epithelial_positive, self.immune_negative):
            semantic_feature = semantic(feature)
            raw.append(torch.cat((
                box(feature),
                objectness(semantic_feature),
                root(semantic_feature),
                epithelial(semantic_feature),
                immune_polarity(semantic_feature),
            ), dim=1))

        if self.training:
            return raw

        self.anchors, self.strides = (
            tensor.transpose(0, 1) for tensor in make_anchors(raw, self.stride)
        )
        flattened = torch.cat(
            [tensor.view(raw[0].shape[0], self.no, -1) for tensor in raw], dim=2
        )
        box, objectness, root, epithelial, immune_polarity = flattened.split(
            (4 * self.ch, 1, 3, 1, 1), dim=1
        )
        left_top, right_bottom = self.dfl(box).chunk(2, 1)
        left_top = self.anchors.unsqueeze(0) - left_top
        right_bottom = self.anchors.unsqueeze(0) + right_bottom
        decoded_box = torch.cat(
            ((left_top + right_bottom) / 2, right_bottom - left_top), dim=1
        )
        class_probabilities = self.probabilities(
            objectness, root, epithelial, immune_polarity, dim=1
        )
        predictions = torch.cat(
            (decoded_box * self.strides, class_probabilities), dim=1
        )
        if not return_components:
            return predictions
        return {
            'predictions': predictions,
            'objectness': objectness.sigmoid(),
            'root_probability': root.softmax(dim=1),
            'epithelial_positive_probability': epithelial.sigmoid(),
            'immune_negative_probability': immune_polarity.sigmoid(),
        }

    def initialize_biases(self):
        for box, objectness, root, epithelial, immune_polarity, stride in zip(
                self.box, self.objectness, self.root, self.epithelial_positive,
                self.immune_negative, self.stride):
            box[-1].bias.data[:] = 1.0
            objectness.bias.data[:] = math.log(5 / (640 / stride) ** 2)
            root.bias.data.zero_()
            epithelial.bias.data.zero_()
            immune_polarity.bias.data.zero_()


class PDL1MergedImmuneYOLO(PDL1HierarchicalYOLO):
    def __init__(self, width, depth, csp):
        torch.nn.Module.__init__(self)
        self.net = DarkNet(width, depth, csp)
        self.fpn = DarkFPN(width, depth, csp)
        img_dummy = torch.zeros(1, width[0], 256, 256)
        self.head = PDL1MergedImmuneHead((width[3], width[4], width[5]))
        self.head.stride = torch.tensor(
            [256 / output.shape[-2] for output in self.forward(img_dummy)]
        )
        self.stride = self.head.stride
        self.head.initialize_biases()



def yolo_v11_m_pdl1_merged_immune():
    return PDL1MergedImmuneYOLO(
        [3, 64, 128, 256, 512, 512], [1, 1, 1, 1, 1, 1], [True, True]
    )


class PDL1MergedImmuneLoss(PDL1HierarchicalComputeLoss):
    """Root and polarity losses masked to their corresponding cell groups."""
    ROOT_TARGET = (0, 1, 0, 1, 2)
    EPITHELIAL_CLASSES = (0, 2)
    IMMUNE_CLASSES = (1, 3)

    def __init__(self, model, params: Mapping[str, Any] | None = None):
        if hasattr(model, 'module'):
            model = model.module
        head = model.head
        if getattr(head, 'hierarchy_name', None) != 'pdl1_merged_immune':
            raise TypeError(
                'PDL1MergedImmuneLoss requires a PDL1MergedImmuneHead model'
            )

        self.params = dict(params or {})
        self.device = next(model.parameters()).device
        self.stride = head.stride
        self.nc = head.nc
        self.no = head.no
        self.reg_max = head.ch
        self.box_loss = util.BoxLoss(head.ch - 1).to(self.device)
        self.assigner = util.Assigner(
            nc=self.nc,
            top_k=int(self.params.get('top_k', 10)),
            alpha=float(self.params.get('assigner_alpha', 0.5)),
            beta=float(self.params.get('assigner_beta', 6.0)),
        )
        self.project = torch.arange(head.ch, dtype=torch.float, device=self.device)
        self.root_target = torch.tensor(
            self.ROOT_TARGET, dtype=torch.long, device=self.device
        )

        self.box_gain = float(self.params.get('box', 5.0))
        self.dfl_gain = float(self.params.get('dfl', 0.5))
        self.objectness_gain = float(self.params.get('objectness', 1.0))
        self.root_gain = float(self.params.get('root', 1.0))
        self.epithelial_gain = float(self.params.get('epithelial', 1.0))
        self.immune_polarity_gain = float(
            self.params.get('immune_polarity', 1.0)
        )
        self.objectness_focal_gamma = float(
            self.params.get('objectness_focal_gamma', 1.5)
        )
        self.root_class_weights = torch.as_tensor(
            self.params.get('root_class_weights', (1.0, 1.0, 1.0)),
            dtype=torch.float32, device=self.device,
        )
        if self.root_class_weights.numel() != 3:
            raise ValueError('root_class_weights must have 3 values')
        self.epithelial_positive_weight = float(
            self.params.get('epithelial_positive_weight', 1.0)
        )
        self.immune_negative_weight = float(
            self.params.get('immune_negative_weight', 1.0)
        )

    def __call__(self, outputs, targets):
        flattened = torch.cat(
            [tensor.view(outputs[0].shape[0], self.no, -1) for tensor in outputs],
            dim=2,
        )
        pred_dist, objectness, root, epithelial, immune_polarity = (
            flattened.split((self.reg_max * 4, 1, 3, 1, 1), dim=1)
        )
        pred_dist = pred_dist.permute(0, 2, 1).contiguous()
        objectness = objectness.permute(0, 2, 1).contiguous()
        root = root.permute(0, 2, 1).contiguous()
        epithelial = epithelial.permute(0, 2, 1).contiguous().squeeze(-1)
        immune_polarity = immune_polarity.permute(0, 2, 1).contiguous().squeeze(-1)
        data_type = pred_dist.dtype

        anchor_points, stride_tensor = util.make_anchors(
            outputs, self.stride, offset=0.5
        )
        gt = self._prepare_targets(outputs, targets)
        gt_labels, gt_bboxes = gt.split((1, 4), dim=2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        pred_bboxes = self.box_decode(anchor_points, pred_dist)

        with torch.no_grad():
            flat_scores = PDL1MergedImmuneHead.probabilities(
                objectness, root, epithelial.unsqueeze(-1),
                immune_polarity.unsqueeze(-1), dim=-1,
            )
            target_bboxes, target_scores, fg_mask = self.assigner(
                flat_scores,
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )

        target_scores = target_scores.to(data_type)
        target_quality = target_scores.amax(dim=-1)
        target_scores_sum = target_quality.sum().clamp_min(1.0)
        objectness_target = target_quality.unsqueeze(-1)
        objectness_element = F.binary_cross_entropy_with_logits(
            objectness, objectness_target, reduction='none'
        )
        if self.objectness_focal_gamma > 0:
            probability = objectness.sigmoid()
            p_t = (
                objectness_target * probability
                + (1 - objectness_target) * (1 - probability)
            )
            objectness_element *= (1 - p_t).pow(self.objectness_focal_gamma)
        loss_objectness = objectness_element.sum() / target_scores_sum

        zero = torch.zeros(1, device=self.device, dtype=data_type)
        loss_root = zero.clone()
        loss_epithelial = zero.clone()
        loss_immune_polarity = zero.clone()
        loss_box = zero.clone()
        loss_dfl = zero.clone()

        if fg_mask.any():
            assigned_class = target_scores.argmax(dim=-1)
            foreground_class = assigned_class[fg_mask]
            foreground_quality = target_quality[fg_mask]
            root_target = self.root_target[foreground_class]
            root_element = F.cross_entropy(
                root[fg_mask], root_target,
                weight=self.root_class_weights.to(root.dtype), reduction='none',
            )
            loss_root = (
                root_element * foreground_quality
            ).sum() / foreground_quality.sum().clamp_min(1.0)

            epithelial_mask = fg_mask & (
                (assigned_class == 0) | (assigned_class == 2)
            )
            if epithelial_mask.any():
                epithelial_target = (assigned_class[epithelial_mask] == 2).to(data_type)
                loss_epithelial = self._weighted_binary_loss(
                    epithelial[epithelial_mask], epithelial_target,
                    target_quality[epithelial_mask],
                    self.epithelial_positive_weight,
                )

            immune_mask = fg_mask & (
                (assigned_class == 1) | (assigned_class == 3)
            )
            if immune_mask.any():
                immune_class = assigned_class[immune_mask]
                immune_quality = target_quality[immune_mask]
                negative_target = (immune_class == 1).to(data_type)
                loss_immune_polarity = self._weighted_binary_loss(
                    immune_polarity[immune_mask], negative_target, immune_quality,
                    self.immune_negative_weight,
                )

            scaled_target_bboxes = target_bboxes / stride_tensor
            loss_box, loss_dfl = self.box_loss(
                pred_dist, pred_bboxes, anchor_points, scaled_target_bboxes,
                target_scores, target_scores_sum, fg_mask,
            )

        losses = {
            'box': loss_box * self.box_gain,
            'objectness': loss_objectness * self.objectness_gain,
            'root': loss_root * self.root_gain,
            'epithelial': loss_epithelial * self.epithelial_gain,
            'immune_polarity': (
                loss_immune_polarity * self.immune_polarity_gain
            ),
            'dfl': loss_dfl * self.dfl_gain,
        }
        losses['total'] = sum(losses.values())
        return losses
