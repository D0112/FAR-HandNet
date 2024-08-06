# Ultralytics YOLO 🚀, AGPL-3.0 license
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA,OKS_SIGMA_HAND
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

from .metrics import bbox_iou
from .tal import bbox2dist

import torch.distributions as distributions
import numpy as np
from ultralytics.nn.modules.realnvp import RealNVP
import math

def GetSigma(gt_kpts, pre_kpts, area):
    nkpt = gt_kpts.shape[1]
    nbox = gt_kpts.shape[0]
    sigma_all = []
    for j in range(nbox):
        sigma = []
        for i in range(nkpt):
            d = (gt_kpts[j, i, 0] - pre_kpts[:, i, 0]) ** 2 + (gt_kpts[j, i, 1] - pre_kpts[:, i, 1]) ** 2
            r = torch.sqrt(d / area[j])
            simga_i = torch.sqrt(torch.sum((r - torch.mean(r)) ** 2) / nkpt)
            sigma.append(simga_i)
        sigma_all.append(sigma)
    # sigma = torch.tensor(sigma).sigmoid()
    return torch.tensor(sigma_all,device=gt_kpts.device).unsqueeze(2) if nbox > 1 else torch.tensor(sigma_all,device=gt_kpts.device)


class RLELoss(nn.Module):
    ''' RLE Regression Loss'''

    def __init__(self, nets, nett, OUTPUT_3D=False, size_average=True,device='cpu'):
        super(RLELoss, self).__init__()
        self.nets = nets
        self.nett = nett
        self.size_average = size_average
        self.amp = 1 / math.sqrt(2 * math.pi)
        prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
        masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32)).to(device)
        self.flow = RealNVP(nets, nett, masks, prior)

    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)

    def forward(self, pred_kpts, gt_kpts, kpt_mask ,sigma):
        pred_jts = pred_kpts[:,:,:2]
        sigma = sigma
        gt_uv = gt_kpts[:,:,:2].to(pred_jts.dtype)
        gt_uv_weight = kpt_mask
        kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0)) / (torch.sum(kpt_mask != 0) + 1e-9)
        bar_mu = torch.abs(pred_jts - gt_uv) / sigma.to(pred_jts.dtype)
        # bar_mu = (pred_jts - gt_uv)
        # (B, K, 2)
        log_phi = self.flow.log_prob(bar_mu.reshape(-1, 2)).reshape(sigma.shape[0], sigma.shape[1], 1)

        nf_loss = torch.log(sigma) - log_phi

        nf_loss = nf_loss * gt_uv_weight

        residual = True
        if residual:
            Q_logprob = self.logQ(gt_uv, pred_jts, sigma) * gt_uv_weight
            loss = nf_loss + Q_logprob

        if self.size_average and gt_uv_weight.sum() > 0:
            # kpt_loss_factor * ((1 - torch.exp(-e)) * kpt_mask).mean()
            # return loss.sum() / len(loss)
            return loss.mean()
        else:
            return loss.sum()





class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367."""

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction='none') *
                    weight).mean(1).sum()
        return loss


# Losses
class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self, ):
        super().__init__()

    def forward(self, pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)


class KeypointLoss(nn.Module):

    def __init__(self, sigmas) -> None:
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]) ** 2 + (pred_kpts[..., 1] - gt_kpts[..., 1]) ** 2
        kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0)) / (torch.sum(kpt_mask != 0) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / (2  * self.sigmas) ** 2 / (area + 1e-9) / 2  # from cocoeval
        return kpt_loss_factor * ((1 - torch.exp(-e)) * kpt_mask).mean()


# Criterion class for computing Detection training losses
class v8DetectionLoss:

    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


# Criterion class for computing training losses
class v8SegmentationLoss(v8DetectionLoss):

    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.nm = model.model[-1].nm  # number of masks
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch['batch_idx'].view(-1, 1)
            targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError('ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n'
                            "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                            "i.e. 'yolo train model=yolov8n-seg.pt data=coco128.yaml'.\nVerify your dataset is a "
                            "correctly formatted 'segment' dataset using 'data=coco128-seg.yaml' "
                            'as an example.\nSee https://docs.ultralytics.com/tasks/segment/ for help.') from e

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # bbox loss
            loss[0], loss[3] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                              target_scores, target_scores_sum, fg_mask)
            # masks loss
            masks = batch['masks'].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]

            for i in range(batch_size):
                if fg_mask[i].sum():
                    mask_idx = target_gt_idx[i][fg_mask[i]]
                    if self.overlap:
                        gt_mask = torch.where(masks[[i]] == (mask_idx + 1).view(-1, 1, 1), 1.0, 0.0)
                    else:
                        gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
                    xyxyn = target_bboxes[i][fg_mask[i]] / imgsz[[1, 0, 1, 0]]
                    marea = xyxy2xywh(xyxyn)[:, 2:].prod(1)
                    mxyxy = xyxyn * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=self.device)
                    loss[1] += self.single_mask_loss(gt_mask, pred_masks[i][fg_mask[i]], proto[i], mxyxy, marea)  # seg

                # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
                else:
                    loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box / batch_size  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        """Mask loss for one image."""
        pred_mask = (pred @ proto.view(self.nm, -1)).view(-1, *proto.shape[1:])  # (n, 32) @ (32,80,80) -> (n,80,80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='none')
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean()


# Criterion class for computing training losses
class v8PoseLoss(v8DetectionLoss):

    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx, fg_mask_kpt, target_gt_idx_kpt= self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
            keypoints = batch['keypoints'].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            for i in range(batch_size):
                if fg_mask[i].sum():
                    idx = target_gt_idx[i][fg_mask[i]]
                    gt_kpt = keypoints[batch_idx.view(-1) == i][idx]  # (n, 51)
                    gt_kpt[..., 0] /= stride_tensor[fg_mask[i]]
                    gt_kpt[..., 1] /= stride_tensor[fg_mask[i]]
                    area = xyxy2xywh(target_bboxes[i][fg_mask[i]])[:, 2:].prod(1, keepdim=True)
                    pred_kpt = pred_kpts[i][fg_mask[i]]
                    kpt_mask = gt_kpt[..., 2] != 0
                    loss[1] += self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss
                    # kpt_score loss
                    if pred_kpt.shape[-1] == 3:
                        loss[2] += self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose / batch_size  # pose gain
        loss[2] *= self.hyp.kobj / batch_size  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def kpts_decode(self, anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y


class v8ClassificationLoss:

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = torch.nn.functional.cross_entropy(preds, batch['cls'], reduction='sum') / 64
        loss_items = loss.detach()
        return loss, loss_items



class v8PoseLoss_RLE(v8DetectionLoss):

    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        # sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        # self.get_sigma = Linear(64, nkpt*2, norm=False, device=self.device)
        self.get_sigma = model.model[-4]
        self.nets = model.model[-3]
        self.nett = model.model[-2]
        self.keypoint_rle_loss = RLELoss(self.nets, self.nett, OUTPUT_3D=False, size_average=True,device=self.device)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        anchor_size = stride_tensor.clone()

        # 使用 torch.where 进行条件替换
        anchor_size = torch.where(stride_tensor == 8, torch.tensor(80), anchor_size)
        anchor_size = torch.where(stride_tensor == 16, torch.tensor(40), anchor_size)
        anchor_size = torch.where(stride_tensor == 32, torch.tensor(20), anchor_size)

        # targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx, fg_mask_kpt, target_gt_idx_kpt = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
            keypoints = batch['keypoints'].to(self.device).clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            for i in range(batch_size):
                if fg_mask[i].sum():
                    idx = target_gt_idx[i][fg_mask[i]]
                    gt_kpt = keypoints[batch_idx.view(-1) == i][idx]  # (n, 51)
                    gt_kpt[..., 0] /= stride_tensor[fg_mask[i]]
                    gt_kpt[..., 1] /= stride_tensor[fg_mask[i]]
                    area = xyxy2xywh(target_bboxes[i][fg_mask[i]])[:, 2:].prod(1, keepdim=True)
                    pred_kpt = pred_kpts[i][fg_mask[i]]
                    pred_kpt_sigma=pred_kpt[:,:,:2].reshape(pred_kpt.shape[0],-1)
                    kpt_mask = gt_kpt[..., 2] != 0
                    kpt_mask_rle = kpt_mask.unsqueeze(2)
                    sigma = torch.abs(self.get_sigma(pred_kpt_sigma).reshape(pred_kpt.shape[0], pred_kpt.shape[1],-1)).sigmoid()

                    anchor_size_i=anchor_size[fg_mask[i]]
                    tensor2_expanded = anchor_size_i.unsqueeze(2).expand(-1, 21, 2)
                    sigma = sigma * tensor2_expanded
                    loss[1] += self.keypoint_rle_loss(pred_kpt, gt_kpt, kpt_mask_rle,sigma)  # pose loss
                    # kpt_score loss
                    if pred_kpt.shape[-1] == 3:
                        loss[2] += self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose / batch_size  # pose gain
        loss[2] *= self.hyp.kobj / batch_size  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


    def kpts_decode(self, anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y


class v8PoseLoss_KptAssigner(v8DetectionLoss):
    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.eps = 1e-30
        self.topk = 10
        self.alpha = 1.0
        self.beta = 3.0
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA_HAND).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        anchor_points_view=anchor_points.cpu().numpy()
        # targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx, fg_mask_kpt, target_gt_idx_kpt= self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
            keypoints = batch['keypoints'].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            for i in range(batch_size):
                if fg_mask[i].sum():
                    init_kpt = keypoints[batch_idx.view(-1) == i]
                    edge_len = self.SelectMaxJointDis(init_kpt)
                    edge_len_repeat = edge_len.repeat(21, 1).T
                    init_boxes = self.kpt_boxes(keypoints[batch_idx.view(-1) == i], edge_len_repeat).clamp(0)
                    kpt_surrframe_mask = self.select_candidates_in_kpt_surrframe(anchor_points * stride_tensor,
                                                                                 init_boxes).sum(dim=1).sum(dim=0)
                    kpt_surrframe_mask = fg_mask_kpt[i] | kpt_surrframe_mask.bool()
                    nonzero_num = kpt_surrframe_mask.count_nonzero()
                    if nonzero_num >= self.topk:
                        mask_kpt = kpt_surrframe_mask
                        idx = target_gt_idx_kpt[i][mask_kpt]
                        if torch.any(idx):
                            idx_ = idx.sort()
                            idx = idx_.values

                        gt_kpt = keypoints[batch_idx.view(-1) == i][idx]  # (n, 51)
                        gt_kpt[..., 0] /= stride_tensor[mask_kpt]
                        gt_kpt[..., 1] /= stride_tensor[mask_kpt]
                        pred_kpt = pred_kpts[i][mask_kpt]
                        unique_idx = idx.unique(return_inverse=True, return_counts=True)
                        kpt_boxes, pre_kpt_boxes = self.get_Boxes(gt_kpt, pred_kpt, unique_idx, edge_len,
                                                                  stride_tensor[mask_kpt])
                        align_metric, overlaps, mask_topk = self.cal_metrics(pred_kpt[:, :, 2:], pre_kpt_boxes.clamp(0),
                                                                             kpt_boxes.clamp(0))
                        pred_kpt = pred_kpt[mask_topk.bool()].view(self.topk, pred_kpt.shape[1], pred_kpt.shape[2])
                        gt_kpt = gt_kpt[mask_topk.bool()].view(self.topk, gt_kpt.shape[1], gt_kpt.shape[2])
                        area = xyxy2xywh(target_bboxes[i][mask_kpt][mask_topk.bool()])[:, 2:].prod(1, keepdim=True)
                        kpt_mask = gt_kpt[..., 2] != 0
                        loss[1] += self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss
                        # kpt_score loss
                        if pred_kpt.shape[-1] == 3:
                            loss[2] += self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss
                    else:
                        mask_kpt = fg_mask_kpt[i]
                        idx = target_gt_idx_kpt[i][mask_kpt]
                        gt_kpt = keypoints[batch_idx.view(-1) == i][idx]  # (n, 51)
                        gt_kpt[..., 0] /= stride_tensor[mask_kpt]
                        gt_kpt[..., 1] /= stride_tensor[mask_kpt]
                        area = xyxy2xywh(target_bboxes[i][mask_kpt])[:, 2:].prod(1, keepdim=True)
                        pred_kpt = pred_kpts[i][mask_kpt]
                        kpt_mask = gt_kpt[..., 2] != 0
                        loss[1] += self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss
                        # kpt_score loss
                        if pred_kpt.shape[-1] == 3:
                            loss[2] += self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose / batch_size  # pose gain
        loss[2] *= self.hyp.kobj / batch_size  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
    def select_candidates_in_kpt_surrframe(self, xy_centers, gt_bboxes, eps=1e-9):
        """
        Select the positive anchor center in gt.

        Args:
            xy_centers (Tensor): shape(h*w, 2)
            gt_bboxes (Tensor): shape(b, n_boxes, 4)

        Returns:
            (Tensor): shape(b, n_boxes, h*w)
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
        return bbox_deltas.amin(3).gt_(eps)


    def cal_joint_dis(self, kpt, figure_id_interval):
        dis = []
        for i in range(len(figure_id_interval)):
            for j in range(3):
                dis.append(torch.norm((kpt[figure_id_interval[i][j], :2] - kpt[figure_id_interval[i][j+1], :2])))
        return dis[1:]
    def SelectMaxJointDis(self, kpt):
        # kpt(n,21,3) n为一张图中拼接的图片数量
        n, kpt_num, _ = kpt.shape
        nums = list(range(kpt_num))
        figure_id_interval = [nums[i:i + 4] for i in range(1, len(nums), 4)]
        max_joint_diss = []
        for i in range(n):
            joint_dis = self.cal_joint_dis(kpt[i, :, :], figure_id_interval)
            max_joint_diss.append(max(joint_dis))
        return torch.tensor(max_joint_diss,device=kpt.device)

    def kpts_decode(self, anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def kpt_boxes(self, centers, dis):
        # x = centers[..., 0] - torch.tensor(weight / 2)
        # y = centers[..., 0] - torch.tensor(height / 2)
        # xy = torch.stack([x,y],dim=2)
        # temp = []
        # for j in range(0, centers.shape[0]):
        #     for i in range(0,len(dis)):
        #         lt_xy = torch.stack([centers[j, i, 0] - dis[i] / 2, centers[j, i, 1] - dis[i] / 2],
        #                           dim=0)
        #         rb_xy = torch.stack([centers[j, i, 0] + dis[i] / 2, centers[j, i, 1] + dis[i] / 2],
        #                           dim=0)
        #         lt_rb = torch.cat((lt_xy[0, ...], rb_xy[0, ...]), dim=0)
        #         temp.append(lt_rb)
        # kpt_bboxes = torch.cat(temp, dim=0).view(21, 4)
        # kpt_bboxes = torch.cat([lt_xy, rb_xy],dim=2)
        # dis.to(centers.device())
        x_min = centers[:, :, 0] - dis[:, :] / 2
        y_min = centers[:, :, 1] - dis[:, :] / 2

        # 计算右下角坐标
        x_max = centers[:, :, 0] + dis[:, :] / 2
        y_max = centers[:, :, 1] + dis[:, :] / 2

        # 组合成矩形框张量
        kpt_bboxes = torch.stack([x_min, y_min, x_max, y_max], dim=2)
        return kpt_bboxes

    def cal_metrics(self, pk_scores, pk_bboxes, kgt_bboxes):
        align_metric, overlaps = self.get_pose_box_metrics(pk_scores, pk_bboxes, kgt_bboxes)
        # mask_topk = self.select_topk_candidates_kpt(align_metric)
        mask_topk = self.select_topk_candidates_kpts(align_metric)
        # _, indexs = torch.max(align_metric, dim=0)
        align_metric = align_metric[mask_topk.bool()].view(-1, self.topk)
        return align_metric, overlaps, mask_topk

    def select_topk_candidates_kpts(self, metrics, largest=True, topk_mask=None):
        metrics_sum=metrics.sum(dim=1)
        topk_metrics, topk_idxs = torch.topk(metrics_sum, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape[0], dtype=torch.int8, device=topk_idxs.device)
        # count_tensor = torch.zeros((21,8400), dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[k:k + 1], ones)
        # count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8, device=topk_idxs.device))
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)
    def select_topk_candidates_kpt(self, metrics, largest=True, topk_mask=None):
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics.T, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.T.shape, dtype=torch.int8, device=topk_idxs.device)
        # count_tensor = torch.zeros((21,8400), dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, k:k + 1], ones)
        # count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8, device=topk_idxs.device))
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_pose_box_metrics(self, pk_scores, pk_bboxes, kgt_bboxes, eps=1e-9):
        overlaps = bbox_iou(kgt_bboxes, pk_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0) + eps
        # print(overlaps)
        # flag = overlaps.any()
        align_metric = pk_scores.abs().squeeze().pow(self.alpha) * overlaps.pow(self.beta)
        # x = pk_scores.squeeze().pow(self.alpha)
        # y = overlaps.pow(self.beta)
        return align_metric, overlaps

    def get_Boxes(self, gt_kpt, pred_kpt, unique_idx, edge_len, stride_tensor):
        gt_boxes = torch.zeros(1, gt_kpt.shape[1], 4, device=gt_kpt.device)
        pred_boxes = torch.zeros(1, gt_kpt.shape[1], 4, device=gt_kpt.device)
        start_index = 0
        for i in range(unique_idx[0].shape[0]):
            sub_edge = (edge_len[i].repeat(stride_tensor.shape[0]).unsqueeze(1) / stride_tensor).repeat(1,gt_kpt.shape[1])
            sub_gt_kpt = gt_kpt[start_index:start_index+unique_idx[2][i], ...]
            sub_pred_kpt = pred_kpt[start_index:start_index+unique_idx[2][i], ...]
            gt_boxes = torch.cat((gt_boxes, self.kpt_boxes(sub_gt_kpt, sub_edge[start_index:start_index+unique_idx[2][i], ...])), dim=0)
            pred_boxes = torch.cat((pred_boxes, self.kpt_boxes(sub_pred_kpt, sub_edge[start_index:start_index+unique_idx[2][i], ...])), dim=0)
            start_index = unique_idx[2][i]+start_index
        return gt_boxes[1:, ...], pred_boxes[1:, ...]

class v8PoseLoss_KptAssigner_RLE(v8DetectionLoss):

    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.eps = 1e-30
        self.topk = 10
        self.alpha = 1.0
        self.beta = 3.0
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        # sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        # self.get_sigma = Linear(64, nkpt*2, norm=False, device=self.device)
        self.get_sigma = model.model[-4]
        self.nets = model.model[-3]
        self.nett = model.model[-2]
        self.keypoint_rle_loss = RLELoss(self.nets, self.nett, OUTPUT_3D=False, size_average=True,device=self.device)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        anchor_size = stride_tensor.clone()

        # targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx, fg_mask_kpt, target_gt_idx_kpt= self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
            keypoints = batch['keypoints'].to(self.device).clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            for i in range(batch_size):
                if fg_mask[i].sum():
                    init_kpt = keypoints[batch_idx.view(-1) == i]
                    edge_len = self.SelectMaxJointDis(init_kpt)
                    edge_len_repeat = edge_len.repeat(21, 1).T
                    init_boxes = self.kpt_boxes(keypoints[batch_idx.view(-1) == i], edge_len_repeat).clamp(0)
                    kpt_surrframe_mask = self.select_candidates_in_kpt_surrframe(anchor_points * stride_tensor,
                                                                                 init_boxes).sum(dim=1).sum(dim=0)
                    kpt_surrframe_mask = fg_mask_kpt[i] | kpt_surrframe_mask.bool()
                    nonzero_num = kpt_surrframe_mask.count_nonzero()
                    if nonzero_num >= self.topk:
                        mask_kpt = kpt_surrframe_mask
                        # mask_kpt = nonzero_num >= self.topk if kpt_surrframe_mask else fg_mask_kpt[i]
                        idx = target_gt_idx_kpt[i][mask_kpt]
                        # id1 = target_gt_idx_kpt[i][fg_mask_kpt[i]]
                        # idxs = target_gt_idx[i][fg_mask[i]]
                        # unique_idx1 = idx.unique(return_inverse=True, return_counts=True)
                        # unique_idx2 = id1.unique(return_inverse=True, return_counts=True)
                        # unique_idx3 = idxs.unique(return_inverse=True, return_counts=True)

                        if torch.any(idx):
                            idx_ = idx.sort()
                            idx = idx_.values

                        gt_kpt = keypoints[batch_idx.view(-1) == i][idx]  # (n, 51)
                        gt_kpt[..., 0] /= stride_tensor[mask_kpt]
                        gt_kpt[..., 1] /= stride_tensor[mask_kpt]
                        pred_kpt = pred_kpts[i][mask_kpt]

                        # unique返回值[1,1,2,3]：
                        #   1.无重复元素的张量[1,2,3]
                        #   2.对应1中的元素索引[0,0,1,2]
                        #   3.元素重复个数[2,1,1]
                        unique_idx = idx.unique(return_inverse=True, return_counts=True)

                        kpt_boxes, pre_kpt_boxes = self.get_Boxes(gt_kpt, pred_kpt, unique_idx, edge_len,
                                                                  stride_tensor[mask_kpt])
                        align_metric, overlaps, mask_topk = self.cal_metrics(pred_kpt[:, :, 2:], pre_kpt_boxes.clamp(0),
                                                                             kpt_boxes.clamp(0))
                        pred_kpt = pred_kpt[mask_topk.bool()].view(self.topk, pred_kpt.shape[1], pred_kpt.shape[2])
                        gt_kpt = gt_kpt[mask_topk.bool()].view(self.topk, gt_kpt.shape[1], gt_kpt.shape[2])
                        kpt_mask = gt_kpt[..., 2] != 0
                        pred_kpt_sigma = pred_kpt[:, :, :2].reshape(pred_kpt.shape[0], -1)
                        kpt_mask_rle = kpt_mask.unsqueeze(2)
                        sigma = torch.abs(
                            self.get_sigma(pred_kpt_sigma).reshape(pred_kpt.shape[0], pred_kpt.shape[1], -1)).sigmoid()
                        # sigma = self.get_sigma(pred_distri[i][fg_mask[i]]).reshape(pred_kpt.shape[0],pred_kpt.shape[1],-1).sigmoid()
                        # sigma = GetSigma(gt_kpt,pred_kpt,area).sigmoid()
                        loss[1] += self.keypoint_rle_loss(pred_kpt, gt_kpt, kpt_mask_rle, sigma)  # pose loss
                        # kpt_score loss
                        if pred_kpt.shape[-1] == 3:
                            loss[2] += self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss
                    else:
                        mask_kpt = fg_mask_kpt[i]
                        idx = target_gt_idx[i][mask_kpt]
                        gt_kpt = keypoints[batch_idx.view(-1) == i][idx]  # (n, 51)
                        gt_kpt[..., 0] /= stride_tensor[mask_kpt]
                        gt_kpt[..., 1] /= stride_tensor[mask_kpt]
                        area = xyxy2xywh(target_bboxes[i][mask_kpt])[:, 2:].prod(1, keepdim=True)
                        pred_kpt = pred_kpts[i][mask_kpt]
                        pred_kpt_sigma = pred_kpt[:, :, :2].reshape(pred_kpt.shape[0], -1)
                        kpt_mask = gt_kpt[..., 2] != 0
                        kpt_mask_rle = kpt_mask.unsqueeze(2)
                        sigma = torch.abs(
                            self.get_sigma(pred_kpt_sigma).reshape(pred_kpt.shape[0], pred_kpt.shape[1], -1)).sigmoid()
                        anchor_size_i = anchor_size[mask_kpt]
                        tensor2_expanded = anchor_size_i.unsqueeze(2).expand(-1, 21, 2)
                        sigma = sigma * tensor2_expanded
                        loss[1] += self.keypoint_rle_loss(pred_kpt, gt_kpt, kpt_mask_rle, sigma)  # pose loss
                        # kpt_score loss
                        if pred_kpt.shape[-1] == 3:
                            loss[2] += self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose / batch_size  # pose gain
        loss[2] *= self.hyp.kobj / batch_size  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def select_candidates_in_kpt_surrframe(self, xy_centers, gt_bboxes, eps=1e-9):
        """
        Select the positive anchor center in gt.

        Args:
            xy_centers (Tensor): shape(h*w, 2)
            gt_bboxes (Tensor): shape(b, n_boxes, 4)

        Returns:
            (Tensor): shape(b, n_boxes, h*w)
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
        return bbox_deltas.amin(3).gt_(eps)

    def cal_joint_dis(self, kpt, figure_id_interval):
        dis = []
        for i in range(len(figure_id_interval)):
            for j in range(3):
                dis.append(torch.norm((kpt[figure_id_interval[i][j], :2] - kpt[figure_id_interval[i][j + 1], :2])))
        return dis[1:]

    def SelectMaxJointDis(self, kpt):
        # kpt(n,21,3) n为一张图中拼接的图片数量
        n, kpt_num, _ = kpt.shape
        nums = list(range(kpt_num))
        figure_id_interval = [nums[i:i + 4] for i in range(1, len(nums), 4)]
        max_joint_diss = []
        for i in range(n):
            joint_dis = self.cal_joint_dis(kpt[i, :, :], figure_id_interval)
            max_joint_diss.append(max(joint_dis))
        return torch.tensor(max_joint_diss, device=kpt.device)

    def kpts_decode(self, anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def kpt_boxes(self, centers, dis):
        # x = centers[..., 0] - torch.tensor(weight / 2)
        # y = centers[..., 0] - torch.tensor(height / 2)
        # xy = torch.stack([x,y],dim=2)
        # temp = []
        # for j in range(0, centers.shape[0]):
        #     for i in range(0,len(dis)):
        #         lt_xy = torch.stack([centers[j, i, 0] - dis[i] / 2, centers[j, i, 1] - dis[i] / 2],
        #                           dim=0)
        #         rb_xy = torch.stack([centers[j, i, 0] + dis[i] / 2, centers[j, i, 1] + dis[i] / 2],
        #                           dim=0)
        #         lt_rb = torch.cat((lt_xy[0, ...], rb_xy[0, ...]), dim=0)
        #         temp.append(lt_rb)
        # kpt_bboxes = torch.cat(temp, dim=0).view(21, 4)
        # kpt_bboxes = torch.cat([lt_xy, rb_xy],dim=2)
        # dis.to(centers.device())
        x_min = centers[:, :, 0] - dis[:, :] / 2
        y_min = centers[:, :, 1] - dis[:, :] / 2

        # 计算右下角坐标
        x_max = centers[:, :, 0] + dis[:, :] / 2
        y_max = centers[:, :, 1] + dis[:, :] / 2

        # 组合成矩形框张量
        kpt_bboxes = torch.stack([x_min, y_min, x_max, y_max], dim=2)
        return kpt_bboxes

    def cal_metrics(self, pk_scores, pk_bboxes, kgt_bboxes):
        align_metric, overlaps = self.get_pose_box_metrics(pk_scores, pk_bboxes, kgt_bboxes)
        # mask_topk = self.select_topk_candidates_kpt(align_metric)
        mask_topk = self.select_topk_candidates_kpts(align_metric)
        # _, indexs = torch.max(align_metric, dim=0)
        align_metric = align_metric[mask_topk.bool()].view(-1, self.topk)
        return align_metric, overlaps, mask_topk

    def select_topk_candidates_kpts(self, metrics, largest=True, topk_mask=None):
        metrics_sum = metrics.sum(dim=1)
        topk_metrics, topk_idxs = torch.topk(metrics_sum, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape[0], dtype=torch.int8, device=topk_idxs.device)
        # count_tensor = torch.zeros((21,8400), dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[k:k + 1], ones)
        # count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8, device=topk_idxs.device))
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def select_topk_candidates_kpt(self, metrics, largest=True, topk_mask=None):
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics.T, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.T.shape, dtype=torch.int8, device=topk_idxs.device)
        # count_tensor = torch.zeros((21,8400), dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, k:k + 1], ones)
        # count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8, device=topk_idxs.device))
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_pose_box_metrics(self, pk_scores, pk_bboxes, kgt_bboxes, eps=1e-9):
        overlaps = bbox_iou(kgt_bboxes, pk_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0) + eps
        # print(overlaps)
        # flag = overlaps.any()
        align_metric = pk_scores.abs().squeeze().pow(self.alpha) * overlaps.pow(self.beta)
        # x = pk_scores.squeeze().pow(self.alpha)
        # y = overlaps.pow(self.beta)
        return align_metric, overlaps

    def get_Boxes(self, gt_kpt, pred_kpt, unique_idx, edge_len, stride_tensor):
        gt_boxes = torch.zeros(1, gt_kpt.shape[1], 4, device=gt_kpt.device)
        pred_boxes = torch.zeros(1, gt_kpt.shape[1], 4, device=gt_kpt.device)
        start_index = 0
        for i in range(unique_idx[0].shape[0]):
            sub_edge = (edge_len[i].repeat(stride_tensor.shape[0]).unsqueeze(1) / stride_tensor).repeat(1,
                                                                                                        gt_kpt.shape[1])
            sub_gt_kpt = gt_kpt[start_index:start_index + unique_idx[2][i], ...]
            sub_pred_kpt = pred_kpt[start_index:start_index + unique_idx[2][i], ...]
            gt_boxes = torch.cat(
                (gt_boxes, self.kpt_boxes(sub_gt_kpt, sub_edge[start_index:start_index + unique_idx[2][i], ...])),
                dim=0)
            pred_boxes = torch.cat(
                (pred_boxes, self.kpt_boxes(sub_pred_kpt, sub_edge[start_index:start_index + unique_idx[2][i], ...])),
                dim=0)
            start_index = unique_idx[2][i] + start_index
        return gt_boxes[1:, ...], pred_boxes[1:, ...]