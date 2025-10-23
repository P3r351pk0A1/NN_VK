import torch
import torch.nn as nn
import torch.nn.functional as F
from .anchors import iou, cxcywh_to_xyxy


class MultiBoxLoss(nn.Module):
    def __init__(self, priors, iou_threshold=0.5, neg_pos_ratio=3):
        super().__init__()
        self.priors = priors  # cxcywh
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio

    def encode_boxes(self, gt_boxes, priors):
        # gt_boxes, priors: xyxy normalized
        # For simplicity, encode as (gt_cx - pri_cx)/pri_w, etc. (not divided by variance)
        pri_cx = (priors[:, 0])
        pri_cy = (priors[:, 1])
        pri_w = priors[:, 2]
        pri_h = priors[:, 3]
        gx = (gt_boxes[:,0] + gt_boxes[:,2]) / 2
        gy = (gt_boxes[:,1] + gt_boxes[:,3]) / 2
        gw = (gt_boxes[:,2] - gt_boxes[:,0])
        gh = (gt_boxes[:,3] - gt_boxes[:,1])
        return torch.stack([(gx - pri_cx) / pri_w, (gy - pri_cy) / pri_h, torch.log(gw / pri_w), torch.log(gh / pri_h)], dim=1)

    def forward(self, predictions, targets):
        # predictions: (locs B x P x 4, confs B x P x num_classes)
        loc_preds, conf_preds = predictions
        batch_size = conf_preds.shape[0]
        num_priors = conf_preds.shape[1]
        device = conf_preds.device
        priors = self.priors.to(device)

        loc_targets = torch.zeros_like(loc_preds)
        conf_targets = torch.zeros((batch_size, num_priors), dtype=torch.long, device=device)

        for b in range(batch_size):
            gt = targets[b]  # np array Nx5 or tensor
            if isinstance(gt, dict):
                boxes = gt['boxes']
                labels = gt['labels']
            else:
                boxes = torch.tensor(gt[:, :4], device=device)
                labels = torch.tensor(gt[:, 4], dtype=torch.long, device=device)

            if boxes.numel() == 0:
                conf_targets[b] = 0
                continue

            # convert priors to xyxy and gt already assumed xyxy
            pri_xyxy = cxcywh_to_xyxy(priors)
            # ensure boxes and priors normalized
            overlaps = iou(pri_xyxy, boxes)
            best_gt_overlap, best_gt_idx = overlaps.max(dim=1)
            best_prior_overlap, best_prior_idx = overlaps.max(dim=0)

            # ensure each gt has at least one match
            for gt_idx, prior_idx in enumerate(best_prior_idx):
                best_gt_idx[prior_idx] = gt_idx
                best_gt_overlap[prior_idx] = 1.0

            # positive priors
            pos_mask = best_gt_overlap > self.iou_threshold
            assigned_gt = best_gt_idx

            conf_targets[b][~pos_mask] = 0
            conf_targets[b][pos_mask] = labels[assigned_gt[pos_mask]].long()

            # encode box targets for positives
            if pos_mask.any():
                matched_gt_boxes = boxes[assigned_gt[pos_mask]]
                encoded = self.encode_boxes(matched_gt_boxes, priors[pos_mask])
                loc_targets[b][pos_mask] = encoded

        # localization loss
        pos_mask = conf_targets > 0
        num_pos = pos_mask.sum().clamp(min=1).float()
        loc_loss = F.smooth_l1_loss(loc_preds[pos_mask], loc_targets[pos_mask], reduction='sum') / num_pos

        # classification loss (hard negative mining)
        batch_conf = conf_preds.view(-1, conf_preds.size(-1))
        conf_loss_all = F.cross_entropy(batch_conf, conf_targets.view(-1), reduction='none')
        conf_loss_all = conf_loss_all.view(batch_size, num_priors)

        conf_loss_pos = conf_loss_all * (conf_targets > 0).float()
        conf_loss_neg = conf_loss_all.clone()
        conf_loss_neg[conf_targets > 0] = 0

        _, idx = conf_loss_neg.sort(dim=1, descending=True)
        _, orders = idx.sort(dim=1)

        num_neg = (conf_targets > 0).sum(dim=1) * self.neg_pos_ratio
        num_neg = num_neg.clamp(min=1)

        neg_mask = orders < num_neg.unsqueeze(1)

        conf_loss = (conf_loss_pos.sum() + conf_loss_all[neg_mask].sum()) / num_pos

        return loc_loss + conf_loss
