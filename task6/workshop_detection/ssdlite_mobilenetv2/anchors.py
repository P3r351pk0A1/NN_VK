import torch
import math

def generate_ssd_priors(feature_map_shapes, image_size=(320,320), scales=None, aspect_ratios=None):
    """
    feature_map_shapes: list of (H, W) for each feature map
    returns priors: Tensor[num_priors,4] in cx,cy,w,h normalized [0,1]
    """
    if scales is None:
        # default scales roughly similar to SSD-lite
        scales = [0.1, 0.2, 0.35, 0.5]
    if aspect_ratios is None:
        aspect_ratios = [[1.0, 2.0, 0.5]] * len(feature_map_shapes)

    priors = []
    for idx, (fm_h, fm_w) in enumerate(feature_map_shapes):
        scale = scales[min(idx, len(scales)-1)]
        for i in range(fm_h):
            for j in range(fm_w):
                cx = (j + 0.5) / fm_w
                cy = (i + 0.5) / fm_h
                for ar in aspect_ratios[idx]:
                    w = scale * math.sqrt(ar)
                    h = scale / math.sqrt(ar)
                    priors.append([cx, cy, w, h])

    return torch.tensor(priors, dtype=torch.float32)


def cxcywh_to_xyxy(boxes):
    # boxes: N x 4 (cx,cy,w,h)
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def iou(boxes1, boxes2):
    # boxes are xyxy
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # N x M x 2
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # N x M x 2
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / union
