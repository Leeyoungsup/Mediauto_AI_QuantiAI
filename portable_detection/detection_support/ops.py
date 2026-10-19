from time import time
import torch
import torchvision

def wh2xy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else numpy.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def make_anchors(x, strides, offset=0.5):
    assert x is not None
    anchor_tensor, stride_tensor = [], []
    dtype, device = x[0].dtype, x[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_tensor), torch.cat(stride_tensor)

def non_max_suppression(outputs, confidence_threshold=0.001, iou_threshold=0.65,
                        class_agnostic=False, multi_label=False,
                        return_indices=False):
    """Apply NMS to model predictions.

    IHC cell classes are mutually exclusive, so the default is one best class
    per anchor.  Set ``multi_label=True`` only for genuinely multi-label data.
    The previous implementation emitted every class above the threshold, which
    could label the same cell as both ``other`` and a tumor grade.
    """
    max_wh = 7680


    bs = outputs.shape[0]  # batch size
    nc = outputs.shape[1] - 4  # number of classes
    xc = outputs[:, 4:4 + nc].amax(1) > confidence_threshold  # candidates

    # Settings
    start = time()
    limit = 0.5 + 0.05 * bs  # seconds to quit after
    output = [torch.zeros((0, 6), device=outputs.device)] * bs
    output_indices = [torch.zeros(0, dtype=torch.long, device=outputs.device)] * bs
    for index, x in enumerate(outputs):  # image index, image inference
        anchor_indices = xc[index].nonzero(as_tuple=False).squeeze(1)
        x = x.transpose(0, -1)[xc[index]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # matrix nx6 (box, confidence, cls)
        box, cls = x.split((4, nc), 1)
        box = wh2xy(box)  # (cx, cy, w, h) to (x1, y1, x2, y2)
        if nc > 1 and multi_label:
            i, j = (cls > confidence_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
            anchor_indices = anchor_indices[i]
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            confidence_mask = conf.view(-1) > confidence_threshold
            x = torch.cat((box, conf, j.float()), 1)[confidence_mask]
            anchor_indices = anchor_indices[confidence_mask]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        order = x[:, 4].argsort(descending=True)
        x = x[order]  # sort by confidence and remove excess boxes
        anchor_indices = anchor_indices[order]

        # Batched NMS (per-class offset) or class-agnostic NMS
        if class_agnostic:
            c = torch.zeros_like(x[:, 5:6])
        else:
            c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes, scores
        indices = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS

        output[index] = x[indices]
        output_indices[index] = anchor_indices[indices]
        if (time() - start) > limit:
            break  # time limit exceeded

    if return_indices:
        return output, output_indices
    return output
