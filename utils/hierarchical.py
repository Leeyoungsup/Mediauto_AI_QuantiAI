"""Training and checkpoint helpers for hierarchical IHC cell detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from utils import util


def apply_tumor_gate(components, tumor_threshold=0.5):
    """Rebuild predictions with an explicit, calibratable tumor decision.

    Obtain ``components`` with ``model(images, return_components=True)``.  A
    cell below the threshold can only become ``other``; a cell at or above it
    can only become one of the four tumor grades.
    """
    if not 0.0 <= tumor_threshold <= 1.0:
        raise ValueError('tumor_threshold must be between 0 and 1')
    predictions = components['predictions']
    objectness = components['objectness']
    tumor_probability = components['tumor_probability']
    grade_probability = components['grade_probability']
    tumor_mask = tumor_probability >= tumor_threshold

    tumor_scores = objectness * tumor_probability * grade_probability
    tumor_scores = tumor_scores * tumor_mask.to(tumor_scores.dtype)
    other_score = objectness * (1.0 - tumor_probability)
    other_score = other_score * (~tumor_mask).to(other_score.dtype)
    return torch.cat((predictions[:, :4], tumor_scores, other_score), dim=1)


class HierarchicalComputeLoss:
    """Loss for cell objectness -> tumor gate -> tumor grade prediction.

    Dataset labels keep the existing convention: grades 0..3 are tumor cells
    and class 4 is ``other``.  Grade loss is evaluated only on tumor cells.
    ``false_tumor_weight`` increases the cost of routing a non-tumor cell into
    the tumor branch, directly targeting the clinically undesirable error.
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
            flat_scores = torch.cat((
                cell_probability * tumor_probability * grade_probability,
                cell_probability * (1.0 - tumor_probability),
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
            foreground_quality = target_quality[fg_mask]
            foreground_class = assigned_class[fg_mask]
            tumor_target = (foreground_class < self.num_grades).to(data_type)

            tumor_element_loss = F.binary_cross_entropy_with_logits(
                tumor_logits.squeeze(-1)[fg_mask], tumor_target, reduction='none'
            )
            tumor_cost = torch.where(
                tumor_target > 0,
                torch.ones_like(tumor_target),
                torch.full_like(tumor_target, self.false_tumor_weight),
            )
            loss_tumor = (
                tumor_element_loss * foreground_quality * tumor_cost
            ).sum() / target_scores_sum

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


def load_flat_checkpoint_into_hierarchical(
        model, checkpoint: str | Path | Mapping[str, Any], map_location='cpu'):
    """Warm-start a hierarchical model from the existing five-class model.

    Backbone, FPN, box head, and semantic towers are copied.  Grade classifiers
    are initialized from flat classes 0..3, and the tumor gate is initialized
    from the contrast between the mean tumor classifier and ``other``.
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
        if old_weight is None or old_bias is None or old_weight.shape[0] < 5:
            continue

        destination[f'head.grade.{level}.weight'] = old_weight[:4].detach().clone()
        destination[f'head.grade.{level}.bias'] = old_bias[:4].detach().clone()
        destination[f'head.tumor.{level}.weight'] = (
            old_weight[:4].mean(dim=0, keepdim=True) - old_weight[4:5]
        ).detach().clone()
        destination[f'head.tumor.{level}.bias'] = (
            old_bias[:4].mean().view(1) - old_bias[4:5]
        ).detach().clone()

        # At low probabilities, log-sum-exp of the old class biases is a useful
        # approximation of an all-cell objectness prior.
        destination[f'head.objectness.{level}.weight'] = (
            old_weight[:5].mean(dim=0, keepdim=True)
        ).detach().clone()
        destination[f'head.objectness.{level}.bias'] = torch.logsumexp(
            old_bias[:5], dim=0, keepdim=True
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
