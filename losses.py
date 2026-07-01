import math
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalTverskyLoss(nn.Module):


    def __init__(
        self,
        alpha:  float = 0.3,
        beta:   float = 0.7,
        gamma:  float = 0.75,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        self.smooth = smooth

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        preds = torch.sigmoid(logits)


        p = preds.view(preds.shape[0], -1)
        t = targets.view(targets.shape[0], -1)


        tp = (p * t).sum(dim=1)
        fp = (p * (1.0 - t)).sum(dim=1)
        fn = ((1.0 - p) * t).sum(dim=1)

        tversky_idx = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )


        focal_tversky = torch.pow((1.0 - tversky_idx).clamp(min=1e-6), self.gamma)
        return focal_tversky.mean()


class BoundaryWeightedDiceLoss(nn.Module):
    def __init__(
        self,
        smooth:          float = 1.0,
        alpha:           float = 2.0,
        boundary_kernel: int   = 5,
        lambda_aux:      float = 0.1,
        ohem_ratio:      float = 0.1,
    ) -> None:
        super().__init__()
        self.smooth          = smooth
        self.alpha           = alpha
        self.boundary_kernel = boundary_kernel
        self.lambda_aux      = lambda_aux
        self.ohem_ratio      = ohem_ratio
        self._pool = nn.MaxPool2d(
            kernel_size=boundary_kernel,
            stride=1,
            padding=boundary_kernel // 2,
        )

    def _compute_boundary_weight(self, mask: Tensor) -> Tensor:
        dilated  = self._pool(mask)
        boundary = (dilated - mask).clamp(0, 1)
        weight   = 1.0 + self.alpha * boundary
        return weight

    def forward(self, logits: Tensor, targets: Tensor) -> Dict[str, Tensor]:
        preds = torch.sigmoid(logits)


        with torch.no_grad():
            weight = self._compute_boundary_weight(targets)


        w_inter  = (weight * preds * targets).view(preds.shape[0], -1)
        w_pred   = (weight * preds).view(preds.shape[0], -1)
        w_gt     = (weight * targets).view(targets.shape[0], -1)

        dice_num  = 2.0 * w_inter.sum(dim=1) + self.smooth
        dice_den  = w_pred.sum(dim=1) + w_gt.sum(dim=1) + self.smooth
        dice_loss = (1.0 - dice_num / dice_den).mean()


        bce_raw = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        B, C, H, W = bce_raw.shape
        n_pixels = H * W
        top_k_pixels = max(1, int(n_pixels * self.ohem_ratio))

        bce_flat = (weight * bce_raw).view(B, -1)

        topk_bce, _ = torch.topk(bce_flat, top_k_pixels, dim=1)

        ohem_bce_loss = topk_bce.mean()

        total = dice_loss + self.lambda_aux * ohem_bce_loss

        return {"dice_loss": dice_loss, "bce_loss": ohem_bce_loss, "total": total}


class PCCPLoss(nn.Module):
    def __init__(
        self,
        num_anchors:          int   = 512,
        num_negatives:        int   = 256,
        entropy_topk:         float = 0.10,
        conf_threshold:       float = 0.80,
        temperature_infonce:  float = 0.07,
        lambda_cps:           float = 1.0,
    ) -> None:
        super().__init__()
        self.num_anchors    = num_anchors
        self.num_negatives  = num_negatives
        self.entropy_topk   = entropy_topk
        self.conf_threshold = conf_threshold
        self.tau            = temperature_infonce
        self.lambda_cps     = lambda_cps

    def _compute_binary_entropy(self, prob: Tensor) -> Tensor:
        eps = 1e-8
        p   = prob.clamp(eps, 1 - eps)
        h   = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
        return h

    def _sample_anchors(self, entropy: Tensor) -> Tensor:
        H, W = entropy.shape[-2:]
        L    = H * W
        ent_flat = entropy.view(-1)

        k = max(self.num_anchors, int(L * self.entropy_topk))
        _, top_indices = torch.topk(ent_flat, k=k)

        perm = torch.randperm(k, device=entropy.device)[:self.num_anchors]
        anchor_idx = top_indices[perm]
        return anchor_idx

    def _compute_cps_loss(
        self,
        logits_strong: Tensor,
        pred_weak:     Tensor,
    ) -> Tensor:
        B, _, H, W = logits_strong.shape

        conf_mask = (pred_weak > self.conf_threshold) | \
                    (pred_weak < (1.0 - self.conf_threshold))

        if conf_mask.sum() == 0:
            return torch.tensor(0.0, device=logits_strong.device)

        pseudo_label = (pred_weak > 0.5).float()

        loss = F.binary_cross_entropy_with_logits(
            logits_strong, pseudo_label, reduction="none"
        )
        loss = (loss * conf_mask.float()).sum() / (conf_mask.float().sum() + 1e-8)
        return loss

    def _compute_infonce_loss(
        self,
        feat_strong:  Tensor,
        feat_weak:    Tensor,
        pred_weak:    Tensor,
        entropy_weak: Tensor,
    ) -> Tensor:
        device = feat_strong.device
        C      = feat_strong.shape[1]
        H, W   = feat_strong.shape[-2:]
        L      = H * W

        anchor_idx = self._sample_anchors(entropy_weak)
        A = len(anchor_idx)

        fs_flat = feat_strong[0].view(C, L).T.contiguous()
        fw_flat = feat_weak[0].view(C, L).T.contiguous()

        label_flat = (pred_weak[0, 0].view(L) > 0.5).long()

        v_q   = fs_flat[anchor_idx]
        v_pos = fw_flat[anchor_idx]
        q_labels = label_flat[anchor_idx]

        pool_for_cls0 = torch.where(label_flat == 1)[0]
        pool_for_cls1 = torch.where(label_flat == 0)[0]

        cls0_mask = (q_labels == 0)
        cls1_mask = (q_labels == 1)

        if len(pool_for_cls0) < self.num_negatives:
            cls0_mask = torch.zeros_like(cls0_mask)
        if len(pool_for_cls1) < self.num_negatives:
            cls1_mask = torch.zeros_like(cls1_mask)

        if not cls0_mask.any() and not cls1_mask.any():
            return torch.tensor(0.0, device=device)

        M = self.num_negatives
        n0 = int(cls0_mask.sum())
        n1 = int(cls1_mask.sum())

        v_neg = torch.zeros(A, M, C, device=device)

        if n0 > 0:
            ri = torch.randint(len(pool_for_cls0), (n0 * M,), device=device)
            neg_idx = pool_for_cls0[ri]
            v_neg[cls0_mask] = fw_flat[neg_idx].view(n0, M, C)

        if n1 > 0:
            ri = torch.randint(len(pool_for_cls1), (n1 * M,), device=device)
            neg_idx = pool_for_cls1[ri]
            v_neg[cls1_mask] = fw_flat[neg_idx].view(n1, M, C)

        valid_mask = cls0_mask | cls1_mask
        if not valid_mask.any():
            return torch.tensor(0.0, device=device)

        v_q_valid   = v_q[valid_mask]
        v_pos_valid = v_pos[valid_mask]
        v_neg_valid = v_neg[valid_mask]

        sim_pos = (v_q_valid * v_pos_valid).sum(dim=-1) / self.tau

        sim_neg = torch.bmm(
            v_q_valid.unsqueeze(1),
            v_neg_valid.permute(0, 2, 1),
        ).squeeze(1) / self.tau

        logits_info = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)
        labels_info = torch.zeros(v_q_valid.shape[0], dtype=torch.long, device=device)

        return F.cross_entropy(logits_info, labels_info)

    def forward(
        self,
        out_weak:   Dict,
        out_strong: Dict,
    ) -> Dict[str, Tensor]:
        pred_weak    = out_weak["pred"].detach()
        logits_strong = out_strong["logits"]

        loss_cps = self._compute_cps_loss(logits_strong, pred_weak)

        B = pred_weak.shape[0]
        entropy_weak = self._compute_binary_entropy(pred_weak)

        infonce_losses: List[Tensor] = []
        for b in range(B):
            feat_s   = out_strong["features"][b:b+1]
            feat_w   = out_weak["features"][b:b+1]
            pred_w_b = pred_weak[b:b+1]
            ent_w_b  = entropy_weak[b:b+1]

            l_info = self._compute_infonce_loss(feat_s, feat_w, pred_w_b, ent_w_b)
            infonce_losses.append(l_info)

        loss_infonce = torch.stack(infonce_losses).mean()

        loss_total = self.lambda_cps * loss_cps + loss_infonce

        return {
            "cps":     loss_cps,
            "infonce": loss_infonce,
            "total":   loss_total,
        }


class CCDiceLoss(nn.Module):
    def __init__(self, bbox_pad: int = 16, smooth: float = 1e-6) -> None:
        super().__init__()
        self.bbox_pad = bbox_pad
        self.smooth   = smooth

    def _component_dice_single(
        self,
        pred: Tensor,
        gt:   Tensor,
    ) -> Tensor:
        H, W   = gt.shape
        gt_np  = gt.detach().cpu().numpy().astype(np.uint8)
        device = pred.device

        if gt_np.sum() == 0:
            return torch.tensor(0.0, device=device, dtype=pred.dtype)

        num_labels, labels_np = cv2.connectedComponents(gt_np, connectivity=8)

        if num_labels <= 1:
            return torch.tensor(0.0, device=device, dtype=pred.dtype)

        labels_t = torch.from_numpy(labels_np.astype(np.int32)).to(device)

        component_losses: List[Tensor] = []

        for label_id in range(1, num_labels):
            comp_mask = (labels_t == label_id).float()

            ys, xs = torch.where(comp_mask > 0)
            y0 = max(0,     int(ys.min().item()) - self.bbox_pad)
            y1 = min(H - 1, int(ys.max().item()) + self.bbox_pad)
            x0 = max(0,     int(xs.min().item()) - self.bbox_pad)
            x1 = min(W - 1, int(xs.max().item()) + self.bbox_pad)

            pred_win = pred[y0:y1+1, x0:x1+1]
            gt_win   = comp_mask[y0:y1+1, x0:x1+1]

            inter = (pred_win * gt_win).sum()
            denom = pred_win.sum() + gt_win.sum() + self.smooth
            dice  = (2.0 * inter + self.smooth) / denom

            component_losses.append(1.0 - dice)

        if not component_losses:
            return torch.tensor(0.0, device=device, dtype=pred.dtype)

        return torch.stack(component_losses).mean()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        B = pred.shape[0]
        losses: List[Tensor] = []

        for b in range(B):
            loss_b = self._component_dice_single(
                pred[b, 0], target[b, 0]
            )
            losses.append(loss_b)

        return torch.stack(losses).mean()


class CAFASegNetLoss(nn.Module):


    def __init__(self, cfg_train) -> None:
        super().__init__()
        self.lambda_tversky   = getattr(cfg_train, 'lambda_tversky', 1.0)
        self.lambda_deep_sup3 = getattr(cfg_train, 'lambda_deep_sup3', 0.3)
        self.lambda_deep_sup2 = getattr(cfg_train, 'lambda_deep_sup2', 0.15)
        self.lambda_cc_dice   = getattr(cfg_train, 'lambda_cc_dice',    0.8)
        self.cc_dice_bbox_pad = getattr(cfg_train, 'cc_dice_bbox_pad',  16)


        self.dice_loss = BoundaryWeightedDiceLoss(
            lambda_aux   = getattr(cfg_train, 'lambda_bce', 0.1),
            ohem_ratio   = 0.10,
        )


        self.tversky_loss = FocalTverskyLoss(
            alpha = getattr(cfg_train, 'tversky_alpha', 0.3),
            beta  = getattr(cfg_train, 'tversky_beta',  0.7),
            gamma = 0.75,
        )


        self.cc_dice_loss = CCDiceLoss(bbox_pad=self.cc_dice_bbox_pad)


    def _compute_aux_deep_sup_loss(self, aux_logits: Tensor, targets: Tensor) -> Tensor:
        _, _, Hp, Wp = aux_logits.shape
        targets_small = F.interpolate(
            targets, size=(Hp, Wp), mode='nearest'
        )

        preds_small = torch.sigmoid(aux_logits)
        p = preds_small.view(preds_small.shape[0], -1)
        t = targets_small.view(targets_small.shape[0], -1)
        smooth = 1.0
        inter  = (p * t).sum(dim=1)
        dice_l = (1.0 - (2.0 * inter + smooth) / (p.sum(dim=1) + t.sum(dim=1) + smooth)).mean()

        bce_raw  = F.binary_cross_entropy_with_logits(aux_logits, targets_small, reduction="none")
        p_t      = torch.exp(-bce_raw)
        alpha_t  = 0.75 * targets_small + 0.25 * (1 - targets_small)
        focal    = (alpha_t * (1 - p_t) ** 2 * bce_raw).mean()

        return dice_l + 0.25 * focal

    def supervised_loss(
        self,
        model_out: Dict,
        targets:   Tensor,
        model:     nn.Module,
    ) -> Dict[str, Tensor]:

        bw  = self.dice_loss(model_out["logits"], targets)
        tv  = self.tversky_loss(model_out["logits"], targets)
        l_cc_dice = self.cc_dice_loss(model_out["pred"], targets)

        l_deep_sup = torch.tensor(0.0, device=model_out["logits"].device)
        if "aux_logits3" in model_out:
            l_ds3 = self._compute_aux_deep_sup_loss(model_out["aux_logits3"], targets)
            l_ds2 = self._compute_aux_deep_sup_loss(model_out["aux_logits2"], targets)
            l_deep_sup = self.lambda_deep_sup3 * l_ds3 + self.lambda_deep_sup2 * l_ds2


        preds = torch.sigmoid(model_out["logits"])


        with torch.no_grad():
            dilated_gt = F.max_pool2d(targets, kernel_size=11, stride=1, padding=5)


        false_positives = preds * (1.0 - dilated_gt)


        fp_penalty = (false_positives ** 2).mean()


        l_total = (
            bw["total"]
            + self.lambda_tversky * tv
            + self.lambda_cc_dice * l_cc_dice
            + l_deep_sup
            + 5.0 * fp_penalty
        )

        return {
            "dice_loss":    bw["dice_loss"],
            "tversky_loss": tv,
            "cc_dice_loss": l_cc_dice,
            "bce_loss":     bw["bce_loss"],
            "fp_penalty":   fp_penalty,
            "bw_dice":      bw["total"],
            "deep_sup":     l_deep_sup,
            "total":        l_total,
        }
