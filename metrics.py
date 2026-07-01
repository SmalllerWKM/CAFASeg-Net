import math
import cv2
import numpy as np
import torch
from torch import Tensor
from typing import List, Tuple

_NAN = float("nan")
_INF = float("inf")


def _to_binary_np(x: Tensor, threshold: float = 0.5) -> np.ndarray:


    if isinstance(x, Tensor):
        arr = x.detach().cpu().float().squeeze().numpy()
    else:
        arr = np.asarray(x, dtype=np.float32).squeeze()
    return (arr > threshold).astype(np.uint8)


def compute_dice(
    pred:      Tensor,
    target:    Tensor,
    threshold: float = 0.5,
    smooth:    float = 1e-6,
) -> float:


    p = _to_binary_np(pred,   threshold).astype(np.float32)
    t = _to_binary_np(target, threshold).astype(np.float32)

    sum_p = p.sum()
    sum_t = t.sum()


    if sum_p == 0 and sum_t == 0:
        return 1.0


    if sum_t == 0 and sum_p > 0:
        return _NAN


    if sum_p == 0:
        return 0.0

    inter = (p * t).sum()
    return float((2.0 * inter + smooth) / (sum_p + sum_t + smooth))


def compute_iou(
    pred:      Tensor,
    target:    Tensor,
    threshold: float = 0.5,
    smooth:    float = 1e-6,
) -> float:


    p = _to_binary_np(pred,   threshold).astype(np.float32)
    t = _to_binary_np(target, threshold).astype(np.float32)

    sum_p = p.sum()
    sum_t = t.sum()

    if sum_p == 0 and sum_t == 0:
        return 1.0

    if sum_t == 0 and sum_p > 0:
        return _NAN

    if sum_p == 0:
        return 0.0

    inter = (p * t).sum()
    union = (p + t).clip(0, 1).sum()
    return float((inter + smooth) / (union + smooth))


def compute_hd95(
    pred:      Tensor,
    target:    Tensor,
    spacing:   Tuple[float, float] = (1.0, 1.0),
    threshold: float = 0.5,
) -> float:


    try:
        from scipy.spatial import cKDTree
    except ImportError:
        raise ImportError("HD95 requires scipy: pip install scipy")

    p = _to_binary_np(pred,   threshold)
    t = _to_binary_np(target, threshold)

    pred_pts   = np.argwhere(p == 1).astype(np.float32)
    target_pts = np.argwhere(t == 1).astype(np.float32)

    sp = np.array(spacing, dtype=np.float32)
    pred_pts   *= sp
    target_pts *= sp

    if len(pred_pts) == 0 and len(target_pts) == 0:
        return _NAN

    if len(pred_pts) == 0 or len(target_pts) == 0:
        return _INF

    tree_t = cKDTree(target_pts)
    tree_p = cKDTree(pred_pts)
    d_p2t, _ = tree_t.query(pred_pts)
    d_t2p, _ = tree_p.query(target_pts)

    return float(max(np.percentile(d_p2t, 95), np.percentile(d_t2p, 95)))


def compute_component_recall(
    pred:           Tensor,
    target:         Tensor,
    threshold:      float = 0.5,
    iou_threshold:  float = 0.1,
) -> Tuple[float, int, int]:


    p_bin = _to_binary_np(pred,   threshold)
    t_bin = _to_binary_np(target, threshold)

    if t_bin.sum() == 0:
        return float("nan"), 0, 0


    num_labels, labels_np = cv2.connectedComponents(t_bin, connectivity=8)
    n_total = num_labels - 1

    if n_total == 0:
        return float("nan"), 0, 0

    n_hit = 0
    for label_id in range(1, num_labels):
        comp_mask = (labels_np == label_id).astype(np.uint8)

        inter = float((p_bin & comp_mask).sum())
        union = float((p_bin | comp_mask).sum())
        if union < 1:
            continue
        iou = inter / union
        if iou >= iou_threshold:
            n_hit += 1

    recall = n_hit / n_total if n_total > 0 else float("nan")
    return recall, n_hit, n_total


class SegMetrics:


    def __init__(
        self,
        threshold:  float = 0.5,
        spacing:    Tuple[float, float] = (1.0, 1.0),
        compute_hd: bool = True,
    ) -> None:
        self.threshold  = threshold
        self.spacing    = spacing
        self.compute_hd = compute_hd
        self.reset()

    def reset(self) -> None:
        self._dice: List[float] = []
        self._iou:  List[float] = []
        self._hd95: List[float] = []
        self.n_total    = 0
        self.n_tn       = 0
        self.n_fp_skip  = 0
        self.n_inf      = 0

        self._comp_hit   = 0
        self._comp_total = 0

    def update(self, pred: Tensor, target: Tensor) -> None:


        B = pred.shape[0]
        for b in range(B):
            p_b = pred[b]
            t_b = target[b]
            self.n_total += 1

            dice = compute_dice(p_b, t_b, self.threshold)
            iou  = compute_iou( p_b, t_b, self.threshold)


            if math.isnan(dice):
                self.n_fp_skip += 1
                continue


            t_sum = _to_binary_np(t_b, self.threshold).sum()
            p_sum = _to_binary_np(p_b, self.threshold).sum()
            if t_sum == 0 and p_sum == 0:
                self.n_tn += 1

            self._dice.append(dice)
            self._iou.append(iou)


            t_sum_check = _to_binary_np(t_b, self.threshold).sum()
            if t_sum_check > 0:
                _, n_hit, n_comp = compute_component_recall(
                    p_b, t_b, self.threshold
                )
                self._comp_hit   += n_hit
                self._comp_total += n_comp

            if self.compute_hd:
                hd = compute_hd95(p_b, t_b, self.spacing, self.threshold)
                if math.isinf(hd):
                    self.n_inf += 1
                elif not math.isnan(hd):
                    self._hd95.append(hd)

    def compute(self) -> dict:


        def _stats(lst: list) -> Tuple[float, float]:
            if not lst:
                return 0.0, 0.0
            a = np.array(lst, dtype=np.float64)
            return float(a.mean()), float(a.std())

        d_m, d_s = _stats(self._dice)
        i_m, i_s = _stats(self._iou)
        h_m, h_s = _stats(self._hd95)

        comp_recall = (
            self._comp_hit / self._comp_total
            if self._comp_total > 0 else float("nan")
        )

        return {
            "dice":          d_m, "dice_std":  d_s,
            "iou":           i_m, "iou_std":   i_s,
            "hd95":          h_m, "hd95_std":  h_s,
            "comp_recall":   comp_recall,
            "comp_hit":      self._comp_hit,
            "comp_total":    self._comp_total,
            "n_total":       self.n_total,
            "n_valid":       len(self._dice),
            "n_tn":          self.n_tn,
            "n_fp_skip":     self.n_fp_skip,
            "n_inf":         self.n_inf,
            "n_skip":        self.n_fp_skip,
        }

    def __repr__(self) -> str:
        r = self.compute()
        cr = r['comp_recall']
        cr_str = f"{cr:.4f}" if not math.isnan(cr) else "N/A"
        return (
            f"[Foreground metrics]  "
            f"Dice={r['dice']:.4f}±{r['dice_std']:.4f}  "
            f"IoU={r['iou']:.4f}±{r['iou_std']:.4f}  "
            f"HD95={r['hd95']:.2f}±{r['hd95_std']:.2f}mm  "
            f"CompRecall={cr_str}({r['comp_hit']}/{r['comp_total']})  "
            f"(valid={r['n_valid']}/{r['n_total']}, "
            f"TN={r['n_tn']}, FP_skip={r['n_fp_skip']}, inf={r['n_inf']})"
        )


if __name__ == "__main__":
    print("=" * 60)
    print("Foreground metrics self-test")
    print("=" * 60)

    H, W = 64, 64

    def _make_pred(fg_frac: float) -> Tensor:
        t = torch.zeros(1, 1, H, W)
        n = int(H * W * fg_frac)
        t.view(-1)[torch.randperm(H * W)[:n]] = 0.95
        return t

    def _make_gt(fg_frac: float) -> Tensor:
        t = torch.zeros(1, 1, H, W)
        n = int(H * W * fg_frac)
        t.view(-1)[torch.randperm(H * W)[:n]] = 1.0
        return t

    gt   = _make_gt(0.10)
    zero = torch.zeros(1, 1, H, W)

    cases = [
        ("Perfect prediction",      gt * 0.95,        gt,   "Dice≈1.0"),
        ("Missed foreground (pred=0)",  zero,             gt,   "Dice=0.0"),
        ("False positive (GT=0)",    _make_pred(0.05), zero, "Dice=nan -> skipped"),
        ("All background (TN)",    zero,             zero, "Dice=1.0"),
        ("Random prediction",      _make_pred(0.08), gt,   "Dice∈(0,1)"),
    ]

    for name, pred, target, expect in cases:
        d = compute_dice(pred, target)
        i = compute_iou(pred,  target)
        d_str = f"{d:.4f}" if not math.isnan(d) else "nan(skip)"
        i_str = f"{i:.4f}" if not math.isnan(i) else "nan(skip)"
        print(f"  [{name:14s}]  Dice={d_str}  IoU={i_str}  expected: {expect}")

    print()
    metrics = SegMetrics(compute_hd=False)
    metrics.update(_make_pred(0.10), gt)
    metrics.update(zero,             zero)
    metrics.update(zero,             gt)
    metrics.update(_make_pred(0.05), zero)
    print(metrics)

    r = metrics.compute()
    assert r["n_fp_skip"] == 1, f"n_fp_skip should be 1, got {r['n_fp_skip']}"
    assert r["n_tn"]      == 1, f"n_tn should be 1, got {r['n_tn']}"
    assert r["n_valid"]   == 3, f"n_valid should be 3, got {r['n_valid']}"
    assert r["n_total"]   == 4, f"n_total should be 4, got {r['n_total']}"

    assert 0.0 < r["dice"] < 1.0, f"mean should be in (0, 1), got {r['dice']}"

    assert compute_dice(zero, zero)             == 1.0,   "TN should return 1.0"
    assert compute_dice(zero, gt)               == 0.0,   "missed foreground should return 0.0"
    assert math.isnan(compute_dice(_make_pred(0.05), zero)), "FP should return nan"
    print("All assertions passed.")
    print("=" * 60)
