"""Training and checkpoint helpers for hierarchical IHC cell detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from utils import util


def apply_tumor_gate(components, tumor_threshold=0.5):
    """Apply an epithelial/other gate without suppressing immune classes.

    ``gate_class_ids`` are suppressed on the other side of the gate. Classes
    not listed there remain available for both decisions (PD-L1 lymphocytes and
    macrophages). Missing metadata retains the legacy all-target gate.
    """
    if not 0.0 <= tumor_threshold <= 1.0:
        raise ValueError('tumor_threshold must be between 0 and 1')
    predictions = components['predictions']
    objectness = components['objectness']
    tumor_probability = components['tumor_probability']
    grade_probability = components['grade_probability']
    tumor_mask = tumor_probability >= tumor_threshold
    num_grades = grade_probability.shape[1]
    gate_class_ids = tuple(components.get('gate_class_ids', range(num_grades)))

    tumor_scores = objectness * grade_probability
    if gate_class_ids:
        gated_scores = (
            tumor_scores[:, list(gate_class_ids)]
            * tumor_probability.to(tumor_scores.dtype)
            * tumor_mask.to(tumor_scores.dtype)
        )
        tumor_scores = tumor_scores.clone()
        tumor_scores[:, list(gate_class_ids)] = gated_scores
    other_score = objectness * (1.0 - tumor_probability)
    other_score = other_score * (~tumor_mask).to(other_score.dtype)
    return torch.cat((predictions[:, :4], tumor_scores, other_score), dim=1)


class HierarchicalComputeLoss:
    """Loss for cell detection, a selective gate, and detailed classification.

    The gate is supervised only for ``gate_class_ids`` versus ``other``.
    Ungated detailed classes still receive classification loss but never
    receive epithelial/other gate loss.
    """

    def __init__(self, model, params: Mapping[str, Any] | None = None):
        if hasattr(model, 'module'):
            model = model.module
        head = model.head
        if not hasattr(head, 'num_grades'):
            raise TypeError('HierarchicalComputeLoss requires a HierarchicalHead model')

        self.params = dict(params or {})
        self.device = next(model.parameters()).device
        self.stride = head.stride
        self.nc = head.nc
        self.num_grades = head.num_grades
        self.gate_class_ids = tuple(head.gate_class_ids)
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

        self.box_gain = float(self.params.get('box', 7.5))
        self.dfl_gain = float(self.params.get('dfl', 0.5))
        self.objectness_gain = float(self.params.get('objectness', 1.0))
        self.tumor_gain = float(self.params.get('tumor', 1.0))
        self.grade_gain = float(self.params.get('grade', 1.0))
        self.false_tumor_weight = float(self.params.get('false_tumor_weight', 2.0))

    def box_decode(self, anchor_points, pred_dist):
        batch, anchors, channels = pred_dist.shape
        pred_dist = pred_dist.view(batch, anchors, 4, channels // 4)
        pred_dist = pred_dist.softmax(3).matmul(self.project.type(pred_dist.dtype))
        left_top, right_bottom = pred_dist.chunk(2, -1)
        return torch.cat((anchor_points - left_top, anchor_points + right_bottom), dim=-1)

    def _prepare_targets(self, outputs, targets):
        batch_size = outputs[0].shape[0]
        input_size = (
            torch.tensor(outputs[0].shape[2:], device=self.device, dtype=torch.float32)
            * self.stride[0]
        )
        indices = targets['idx'].view(-1, 1)
        classes = targets['cls'].view(-1, 1)
        boxes = targets['box']
        packed = torch.cat((indices, classes, boxes), dim=1).to(self.device)

        if packed.shape[0] == 0:
            return torch.zeros(batch_size, 0, 5, device=self.device, dtype=torch.float32)

        image_indices = packed[:, 0]
        counts = torch.bincount(image_indices.long(), minlength=batch_size)
        gt = torch.zeros(
            batch_size, int(counts.max().item()), 5, device=self.device, dtype=torch.float32
        )
        for image_index in range(batch_size):
            matches = image_indices == image_index
            count = int(matches.sum().item())
            if count:
                gt[image_index, :count] = packed[matches, 1:].float()

        xywh = gt[..., 1:5] * input_size[[1, 0, 1, 0]]
        half_wh = xywh[..., 2:4] / 2
        gt[..., 1:3] = xywh[..., 0:2] - half_wh
        gt[..., 3:5] = xywh[..., 0:2] + half_wh
        return gt

    def __call__(self, outputs, targets):
        flattened = torch.cat(
            [tensor.view(outputs[0].shape[0], self.no, -1) for tensor in outputs], dim=2
        )
        pred_dist, objectness_logits, tumor_logits, grade_logits = flattened.split(
            (self.reg_max * 4, 1, 1, self.num_grades), dim=1
        )
        pred_dist = pred_dist.permute(0, 2, 1).contiguous()
        objectness_logits = objectness_logits.permute(0, 2, 1).contiguous()
        tumor_logits = tumor_logits.permute(0, 2, 1).contiguous()
        grade_logits = grade_logits.permute(0, 2, 1).contiguous()
        data_type = pred_dist.dtype

        anchor_points, stride_tensor = util.make_anchors(outputs, self.stride, offset=0.5)
        gt = self._prepare_targets(outputs, targets)
        gt_labels, gt_bboxes = gt.split((1, 4), dim=2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = self.box_decode(anchor_points, pred_dist)
        with torch.no_grad():
            cell_probability = objectness_logits.sigmoid()
            tumor_probability = tumor_logits.sigmoid()
            grade_probability = grade_logits.softmax(dim=-1)
            detail_scores = cell_probability * grade_probability
            if self.gate_class_ids:
                detail_scores = detail_scores.clone()
                detail_scores[..., list(self.gate_class_ids)] *= tumor_probability.to(
                    detail_scores.dtype
                )
            flat_scores = torch.cat((
                detail_scores, cell_probability * (1.0 - tumor_probability),
            ), dim=-1)
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
        loss_objectness = F.binary_cross_entropy_with_logits(
            objectness_logits, objectness_target, reduction='sum'
        ) / target_scores_sum

        loss_tumor = torch.zeros(1, device=self.device, dtype=data_type)
        loss_grade = torch.zeros(1, device=self.device, dtype=data_type)
        loss_box = torch.zeros(1, device=self.device, dtype=data_type)
        loss_dfl = torch.zeros(1, device=self.device, dtype=data_type)

        if fg_mask.any():
            assigned_class = target_scores.argmax(dim=-1)
            is_gated_class = torch.zeros_like(assigned_class, dtype=torch.bool)
            for class_id in self.gate_class_ids:
                is_gated_class |= assigned_class == class_id
            is_other = assigned_class == self.num_grades
            gate_foreground = fg_mask & (is_gated_class | is_other)

            if gate_foreground.any():
                gate_target = is_gated_class[gate_foreground].to(data_type)
                gate_quality = target_quality[gate_foreground]
                tumor_element_loss = F.binary_cross_entropy_with_logits(
                    tumor_logits.squeeze(-1)[gate_foreground],
                    gate_target,
                    reduction='none',
                )
                tumor_cost = torch.where(
                    gate_target > 0,
                    torch.ones_like(gate_target),
                    torch.full_like(gate_target, self.false_tumor_weight),
                )
                loss_tumor = (
                    tumor_element_loss * gate_quality * tumor_cost
                ).sum() / gate_quality.sum().clamp_min(1.0)

            tumor_foreground = fg_mask & (assigned_class < self.num_grades)
            if tumor_foreground.any():
                grade_element_loss = F.cross_entropy(
                    grade_logits[tumor_foreground],
                    assigned_class[tumor_foreground],
                    reduction='none',
                )
                loss_grade = (
                    grade_element_loss * target_quality[tumor_foreground]
                ).sum() / target_scores_sum

            target_bboxes = target_bboxes / stride_tensor
            loss_box, loss_dfl = self.box_loss(
                pred_dist,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        losses = {
            'box': loss_box * self.box_gain,
            'objectness': loss_objectness * self.objectness_gain,
            'tumor': loss_tumor * self.tumor_gain,
            'grade': loss_grade * self.grade_gain,
            'dfl': loss_dfl * self.dfl_gain,
        }
        losses['total'] = sum(losses.values())
        return losses


class PDL1HierarchicalComputeLoss:
    """Masked multi-task loss for the PD-L1 seven-class hierarchy."""

    # Flat class -> root class: epithelial=0, immune=1, other=2.
    ROOT_TARGET = (0, 1, 1, 0, 1, 1, 2)
    EPITHELIAL_CLASSES = (0, 3)
    IMMUNE_CLASSES = (1, 2, 4, 5)

    def __init__(self, model, params: Mapping[str, Any] | None = None):
        if hasattr(model, 'module'):
            model = model.module
        head = model.head
        if getattr(head, 'hierarchy_name', None) != 'pdl1_epithelial_immune_other':
            raise TypeError(
                'PDL1HierarchicalComputeLoss requires a PDL1HierarchicalHead model'
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
        self.immune_type_gain = float(self.params.get('immune_type', 1.0))
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
        self.immune_lymphocyte_weight = float(
            self.params.get('immune_lymphocyte_weight', 1.0)
        )
        self.immune_negative_weight = float(
            self.params.get('immune_negative_weight', 1.0)
        )

    def box_decode(self, anchor_points, pred_dist):
        batch, anchors, channels = pred_dist.shape
        pred_dist = pred_dist.view(batch, anchors, 4, channels // 4)
        pred_dist = pred_dist.softmax(3).matmul(
            self.project.type(pred_dist.dtype)
        )
        left_top, right_bottom = pred_dist.chunk(2, -1)
        return torch.cat(
            (anchor_points - left_top, anchor_points + right_bottom), dim=-1
        )

    def _prepare_targets(self, outputs, targets):
        batch_size = outputs[0].shape[0]
        input_size = (
            torch.tensor(outputs[0].shape[2:], device=self.device, dtype=torch.float32)
            * self.stride[0]
        )
        packed = torch.cat((
            targets['idx'].view(-1, 1),
            targets['cls'].view(-1, 1),
            targets['box'],
        ), dim=1).to(self.device)
        if packed.shape[0] == 0:
            return torch.zeros(
                batch_size, 0, 5, device=self.device, dtype=torch.float32
            )

        image_indices = packed[:, 0]
        counts = torch.bincount(image_indices.long(), minlength=batch_size)
        gt = torch.zeros(
            batch_size, int(counts.max().item()), 5,
            device=self.device, dtype=torch.float32,
        )
        for image_index in range(batch_size):
            matches = image_indices == image_index
            count = int(matches.sum().item())
            if count:
                gt[image_index, :count] = packed[matches, 1:].float()

        xywh = gt[..., 1:5] * input_size[[1, 0, 1, 0]]
        half_wh = xywh[..., 2:4] / 2
        gt[..., 1:3] = xywh[..., 0:2] - half_wh
        gt[..., 3:5] = xywh[..., 0:2] + half_wh
        return gt

    @staticmethod
    def _weighted_binary_loss(logits, target, quality, positive_weight):
        pos_weight = torch.as_tensor(
            positive_weight, dtype=logits.dtype, device=logits.device
        )
        element_loss = F.binary_cross_entropy_with_logits(
            logits, target.to(logits.dtype), pos_weight=pos_weight,
            reduction='none',
        )
        return (element_loss * quality).sum() / quality.sum().clamp_min(1.0)

    def __call__(self, outputs, targets):
        flattened = torch.cat(
            [tensor.view(outputs[0].shape[0], self.no, -1) for tensor in outputs],
            dim=2,
        )
        pred_dist, objectness, root, epithelial, immune_type, immune_polarity = (
            flattened.split((self.reg_max * 4, 1, 3, 1, 1, 1), dim=1)
        )
        pred_dist = pred_dist.permute(0, 2, 1).contiguous()
        objectness = objectness.permute(0, 2, 1).contiguous()
        root = root.permute(0, 2, 1).contiguous()
        epithelial = epithelial.permute(0, 2, 1).contiguous().squeeze(-1)
        immune_type = immune_type.permute(0, 2, 1).contiguous().squeeze(-1)
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
            # Channel-last form of PDL1HierarchicalHead.probabilities.
            cell = objectness.sigmoid()
            root_probability = root.softmax(dim=-1)
            epithelial_probability = epithelial.sigmoid().unsqueeze(-1)
            lymphocyte_probability = immune_type.sigmoid().unsqueeze(-1)
            negative_probability = immune_polarity.sigmoid().unsqueeze(-1)
            epithelial_root = root_probability[..., 0:1]
            immune_root = root_probability[..., 1:2]
            other_root = root_probability[..., 2:3]
            flat_scores = torch.cat((
                cell * epithelial_root * (1 - epithelial_probability),
                cell * immune_root * lymphocyte_probability * negative_probability,
                cell * immune_root * (1 - lymphocyte_probability) * negative_probability,
                cell * epithelial_root * epithelial_probability,
                cell * immune_root * lymphocyte_probability * (1 - negative_probability),
                cell * immune_root * (1 - lymphocyte_probability) * (1 - negative_probability),
                cell * other_root,
            ), dim=-1)
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
        loss_immune_type = zero.clone()
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
                (assigned_class == 0) | (assigned_class == 3)
            )
            if epithelial_mask.any():
                epithelial_target = (assigned_class[epithelial_mask] == 3).to(data_type)
                loss_epithelial = self._weighted_binary_loss(
                    epithelial[epithelial_mask], epithelial_target,
                    target_quality[epithelial_mask],
                    self.epithelial_positive_weight,
                )

            immune_mask = fg_mask & (
                (assigned_class == 1) | (assigned_class == 2)
                | (assigned_class == 4) | (assigned_class == 5)
            )
            if immune_mask.any():
                immune_class = assigned_class[immune_mask]
                immune_quality = target_quality[immune_mask]
                lymphocyte_target = (
                    (immune_class == 1) | (immune_class == 4)
                ).to(data_type)
                negative_target = (
                    (immune_class == 1) | (immune_class == 2)
                ).to(data_type)
                loss_immune_type = self._weighted_binary_loss(
                    immune_type[immune_mask], lymphocyte_target, immune_quality,
                    self.immune_lymphocyte_weight,
                )
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
            'immune_type': loss_immune_type * self.immune_type_gain,
            'immune_polarity': (
                loss_immune_polarity * self.immune_polarity_gain
            ),
            'dfl': loss_dfl * self.dfl_gain,
        }
        losses['total'] = sum(losses.values())
        return losses


def load_flat_checkpoint_into_pdl1_hierarchical(
        model, checkpoint: str | Path | Mapping[str, Any], map_location='cpu'):
    """Warm-start the PD-L1 hierarchy from a flat seven-class checkpoint."""
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(
            checkpoint, map_location=map_location, weights_only=False
        )
    source = checkpoint.get('model_state_dict', checkpoint)
    source = {key.removeprefix('module.'): value for key, value in source.items()}
    destination = model.state_dict()
    loaded = set()

    for key, value in source.items():
        if key in destination and destination[key].shape == value.shape:
            destination[key] = value.detach().clone()
            loaded.add(key)

    for level in range(model.head.nl):
        for layer in range(4):
            source_prefix = f'head.cls.{level}.{layer}.'
            destination_prefix = f'head.semantic.{level}.{layer}.'
            for key, value in source.items():
                if key.startswith(source_prefix):
                    mapped = destination_prefix + key[len(source_prefix):]
                    if mapped in destination and destination[mapped].shape == value.shape:
                        destination[mapped] = value.detach().clone()
                        loaded.add(mapped)

        old_weight = source.get(f'head.cls.{level}.4.weight')
        old_bias = source.get(f'head.cls.{level}.4.bias')
        if old_weight is None or old_bias is None or old_weight.shape[0] < 7:
            continue

        groups = ((0, 3), (1, 2, 4, 5), (6,))
        root_weight = torch.stack([
            old_weight[list(group)].mean(dim=0) for group in groups
        ])
        root_bias = torch.stack([
            torch.logsumexp(old_bias[list(group)], dim=0) for group in groups
        ])
        mappings = {
            f'head.root.{level}.weight': root_weight,
            f'head.root.{level}.bias': root_bias,
            f'head.epithelial_positive.{level}.weight': (
                old_weight[3:4] - old_weight[0:1]
            ),
            f'head.epithelial_positive.{level}.bias': old_bias[3:4] - old_bias[0:1],
            f'head.immune_lymphocyte.{level}.weight': (
                old_weight[[1, 4]].mean(dim=0, keepdim=True)
                - old_weight[[2, 5]].mean(dim=0, keepdim=True)
            ),
            f'head.immune_lymphocyte.{level}.bias': (
                old_bias[[1, 4]].mean().view(1)
                - old_bias[[2, 5]].mean().view(1)
            ),
            f'head.immune_negative.{level}.weight': (
                old_weight[[1, 2]].mean(dim=0, keepdim=True)
                - old_weight[[4, 5]].mean(dim=0, keepdim=True)
            ),
            f'head.immune_negative.{level}.bias': (
                old_bias[[1, 2]].mean().view(1)
                - old_bias[[4, 5]].mean().view(1)
            ),
            f'head.objectness.{level}.weight': old_weight[:7].mean(
                dim=0, keepdim=True
            ),
            f'head.objectness.{level}.bias': torch.logsumexp(
                old_bias[:7], dim=0, keepdim=True
            ),
        }
        for key, value in mappings.items():
            if key in destination and destination[key].shape == value.shape:
                destination[key] = value.detach().clone()
                loaded.add(key)

    model.load_state_dict(destination)
    return {
        'loaded_tensors': len(loaded),
        'total_tensors': len(destination),
        'loaded_fraction': len(loaded) / max(len(destination), 1),
    }


def load_flat_checkpoint_into_hierarchical(
        model, checkpoint: str | Path | Mapping[str, Any], map_location='cpu'):
    """Warm-start a hierarchical model from a compatible flat-class model.

    Backbone, FPN, box head, and semantic towers are copied whenever shapes
    match. Semantic classifiers are transferred only when the source contains
    ``num_grades + 1`` outputs in the same order (target classes, then other).
    """
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location=map_location, weights_only=False)
    source = checkpoint.get('model_state_dict', checkpoint)
    source = {key.removeprefix('module.'): value for key, value in source.items()}
    destination = model.state_dict()
    loaded = set()

    for key, value in source.items():
        if key in destination and destination[key].shape == value.shape:
            destination[key] = value.detach().clone()
            loaded.add(key)

    for level in range(model.head.nl):
        # Copy the feature-producing portion of the old classification tower.
        for layer in range(4):
            source_prefix = f'head.cls.{level}.{layer}.'
            destination_prefix = f'head.semantic.{level}.{layer}.'
            for key, value in source.items():
                if not key.startswith(source_prefix):
                    continue
                mapped = destination_prefix + key[len(source_prefix):]
                if mapped in destination and destination[mapped].shape == value.shape:
                    destination[mapped] = value.detach().clone()
                    loaded.add(mapped)

        old_weight = source.get(f'head.cls.{level}.4.weight')
        old_bias = source.get(f'head.cls.{level}.4.bias')
        target_count = model.head.num_grades
        required_classes = target_count + 1
        if (old_weight is None or old_bias is None
                or old_weight.shape[0] < required_classes):
            continue

        destination[f'head.grade.{level}.weight'] = old_weight[:target_count].detach().clone()
        destination[f'head.grade.{level}.bias'] = old_bias[:target_count].detach().clone()
        gate_indices = list(model.head.gate_class_ids)
        destination[f'head.tumor.{level}.weight'] = (
            old_weight[gate_indices].mean(dim=0, keepdim=True)
            - old_weight[target_count:required_classes]
        ).detach().clone()
        destination[f'head.tumor.{level}.bias'] = (
            old_bias[gate_indices].mean().view(1)
            - old_bias[target_count:required_classes]
        ).detach().clone()

        # At low probabilities, log-sum-exp of the old class biases is a useful
        # approximation of an all-cell objectness prior.
        destination[f'head.objectness.{level}.weight'] = (
            old_weight[:required_classes].mean(dim=0, keepdim=True)
        ).detach().clone()
        destination[f'head.objectness.{level}.bias'] = torch.logsumexp(
            old_bias[:required_classes], dim=0, keepdim=True
        ).detach().clone()
        loaded.update({
            f'head.grade.{level}.weight', f'head.grade.{level}.bias',
            f'head.tumor.{level}.weight', f'head.tumor.{level}.bias',
            f'head.objectness.{level}.weight', f'head.objectness.{level}.bias',
        })

    model.load_state_dict(destination)
    return {
        'loaded_tensors': len(loaded),
        'total_tensors': len(destination),
        'loaded_fraction': len(loaded) / max(len(destination), 1),
    }
