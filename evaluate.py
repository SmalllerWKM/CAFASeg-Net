import argparse
import math
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor


try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch


    plt.rcParams['axes.unicode_minus'] = False

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    warnings.warn("Matplotlib not installed, visualization skipped. pip install matplotlib")


try:
    from tabulate import tabulate
    _HAS_TABULATE = True
except ImportError:
    _HAS_TABULATE = False


try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    warnings.warn("Scipy not installed, HD95 will not be calculated. pip install scipy")

from config    import cfg, require_runtime_assets
from models    import CAFASegNet
from dataset   import build_labeled_loaders
from inference import keep_largest_component_tensor


_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


_C_TP = np.array([ 50, 205,  80], dtype=np.uint8)
_C_FP = np.array([220,  50,  50], dtype=np.uint8)
_C_FN = np.array([255, 185,   0], dtype=np.uint8)


_MEDICAL_CMAP = (
    LinearSegmentedColormap.from_list("med", [
        (0.00, "#0d0221"), (0.20, "#0a3fa0"),
        (0.45, "#00b4d8"), (0.65, "#90e0ef"),
        (0.80, "#f6c90e"), (1.00, "#e63946"),
    ]) if _HAS_MPL else None
)


def _denorm(t: Tensor) -> np.ndarray:

    img = t.cpu().float().numpy().transpose(1, 2, 0)
    img = np.clip(img * _STD + _MEAN, 0, 1)
    return (img * 255).astype(np.uint8)


def _overlay(img: np.ndarray, mask: np.ndarray,
             color: Tuple, alpha: float = 0.42) -> np.ndarray:

    layer = np.zeros_like(img)
    layer[mask > 0] = color
    out = cv2.addWeighted(img, 1 - alpha, layer, alpha, 0)
    cts, _ = cv2.findContours((mask * 255).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cts, -1, (255, 255, 255), 2)
    return out


def _error_map(img: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:


    dark = (img.astype(np.float32) * 0.30).astype(np.uint8)
    tp   = (pred & gt).astype(bool)
    fp   = (pred & ~gt.astype(bool))
    fn   = (~pred.astype(bool) & gt.astype(bool))
    out  = dark.copy()
    out[tp] = _C_TP;  out[fp] = _C_FP;  out[fn] = _C_FN


    H, W = img.shape[:2]
    lh, lw, pad = 14, 54, 4
    for i, (label, color) in enumerate([("TP", _C_TP), ("FP", _C_FP), ("FN", _C_FN)]):
        y0 = H - (3 - i) * (lh + pad) - pad
        x0 = W - lw - pad
        out[y0:y0+lh, x0:x0+lw] = color
        cv2.putText(out, label, (x0+3, y0+lh-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)
    return out


def _prob2heat(prob: np.ndarray) -> np.ndarray:

    if _HAS_MPL:
        rgba = _MEDICAL_CMAP(np.clip(prob, 0, 1))
        return (rgba[:, :, :3] * 255).astype(np.uint8)
    p8  = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(p8, cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_direct_overlay(
    img:  np.ndarray, gt: np.ndarray, pred: np.ndarray,
    dice: float, hd95: float, path: str, fid: str = "", dpi: int = 150,
) -> None:


    if not _HAS_MPL:
        return


    out = img.copy()


    c_gt_contour = (0, 255, 0)
    c_pr_contour = (255, 0, 255)


    cts_gt, _ = cv2.findContours((gt * 255).astype(np.uint8),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cts_gt, -1, c_gt_contour, 2, cv2.LINE_AA)


    cts_pr, _ = cv2.findContours((pred * 255).astype(np.uint8),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cts_pr, -1, c_pr_contour, 2, cv2.LINE_AA)


    H, W = img.shape[:2]
    d_str = f"Dice: {dice:.4f}" if not math.isnan(dice) else "Dice: N/A"
    h_str = f"HD95: {hd95:.2f}" if not math.isnan(hd95) and not math.isinf(hd95) else "HD95: N/A"
    text = f"{d_str}  |  {h_str}"

    cv2.putText(out, text, (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)


    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi, facecolor="#12121f")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01)

    ax.imshow(out)
    ax.set_title(f"Direct Overlay - Frame: {fid}", fontsize=11, color="white", fontweight="bold")
    ax.axis("off")


    legend_elements = [
        Patch(edgecolor="#00FF00", fc="none", lw=2, label="Ground Truth Boundary"),
        Patch(edgecolor="#FF00FF", fc="none", lw=2, label="Predicted Boundary"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              facecolor="#2a2a3e", edgecolor="gray", labelcolor="white",
              framealpha=0.85, shadow=False)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_comparison(
    img:  np.ndarray, gt: np.ndarray, pred: np.ndarray,
    prob: np.ndarray, dice: float, iou: float,
    path: str, fid: str = "", dpi: int = 120,
) -> None:


    if not _HAS_MPL:
        return

    panels = [
        img,
        _overlay(img, gt,   (80, 220,  80)),
        _overlay(img, pred, (80, 160, 255)),
        None,
        _error_map(img, pred, gt),
    ]
    titles = ["Original Image", "Ground Truth", "Prediction", "Confidence Heatmap", "Error Map"]

    fig = plt.figure(figsize=(25, 5), dpi=dpi, facecolor="#1a1a2e")
    gs  = gridspec.GridSpec(1, 5, figure=fig, wspace=0.04,
                            left=0.01, right=0.99, top=0.87, bottom=0.02)

    for col in range(5):
        ax = fig.add_subplot(gs[0, col])
        if col == 3:
            im = ax.imshow(prob, cmap=_MEDICAL_CMAP, vmin=0, vmax=1,
                           interpolation="bilinear")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                                ticks=[0, 0.5, 1.0])
            cbar.ax.tick_params(labelsize=7, colors="white")
            cbar.outline.set_edgecolor("#555")

            cbar.ax.axhline(0.5, color="white", lw=1.2, ls="--", alpha=0.85)
            cbar.ax.text(1.7, 0.5, "0.5", color="white", fontsize=6,
                         va="center", transform=cbar.ax.transAxes)
        else:
            ax.imshow(panels[col], interpolation="bilinear")
        ax.set_title(titles[col], fontsize=10, color="white",
                     fontweight="bold", pad=4)
        ax.axis("off")

    d_s = f"{dice:.4f}" if not math.isnan(dice) else "N/A"
    i_s = f"{iou:.4f}"  if not math.isnan(iou)  else "N/A"
    fig.suptitle(f"Frame: {fid}    Dice = {d_s}    IoU = {i_s}",
                 fontsize=11, color="white", fontweight="bold", y=0.97)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_heatmap(
    img: np.ndarray, prob: np.ndarray, gt: np.ndarray,
    dice: float, path: str, fid: str = "", dpi: int = 110,
) -> None:


    if not _HAS_MPL:
        return

    heat_rgb = _prob2heat(prob)
    blended  = cv2.addWeighted(img, 0.45, heat_rgb, 0.55, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=dpi,
                             facecolor="#12121f")
    fig.subplots_adjust(wspace=0.06, left=0.02, right=0.96,
                        top=0.87, bottom=0.02)

    axes[0].imshow(img);       axes[0].set_title("Original Image",   fontsize=12, color="white", fontweight="bold")
    axes[2].imshow(blended);   axes[2].set_title("Heatmap Overlay",  fontsize=12, color="white", fontweight="bold")

    im = axes[1].imshow(prob, cmap=_MEDICAL_CMAP, vmin=0, vmax=1,
                         interpolation="bilinear")
    axes[1].contour(prob, levels=[0.5],
                    colors=["white"], linewidths=1.5, linestyles="--")
    if gt.sum() > 0:
        axes[1].contour(gt.astype(np.float32), levels=[0.5],
                        colors=["#FFD700"], linewidths=2.0, linestyles="-")

    axes[1].set_title("Confidence Heatmap", fontsize=12, color="white", fontweight="bold")

    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04,
                        ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=8, colors="white")
    cbar.outline.set_edgecolor("white")

    cbar.set_label("Predicted Probability", color="white", fontsize=9)

    axes[1].legend(handles=[
        Patch(edgecolor="white",   fc="none", ls="--", lw=1.5, label="Threshold τ=0.5"),
        Patch(edgecolor="#FFD700", fc="none", ls="-",  lw=2.0, label="GT Contour"),
    ], loc="lower right", fontsize=7,
       facecolor="#2a2a3e", edgecolor="gray", labelcolor="white")

    for ax in axes:
        ax.axis("off")

    d_s = f"{dice:.4f}" if not math.isnan(dice) else "N/A"
    fig.suptitle(f"Frame: {fid}    Dice = {d_s}",
                 fontsize=12, color="white", fontweight="bold", y=0.96)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_summary_grid(
    records:  List[Dict],
    save_dir: str,
    n_worst:  int = 12,
    n_best:   int = 12,
    dpi:      int = 100,
) -> None:


    if not _HAS_MPL or not records:
        return

    def _draw(subset: List[Dict], title: str, out_path: str):
        n = len(subset)
        if n == 0:
            return
        fig, axes = plt.subplots(n, 4, figsize=(4*4, n*3.5),
                                 dpi=dpi, facecolor="#12121f")
        fig.subplots_adjust(wspace=0.04, hspace=0.20,
                            left=0.10, right=0.99, top=0.96, bottom=0.01)
        if n == 1:
            axes = axes[np.newaxis, :]

        for col, ct in enumerate(["Original Image", "Ground Truth", "Prediction", "Heatmap"]):
            axes[0, col].set_title(ct, fontsize=9, color="white",
                                   fontweight="bold", pad=3)

        for row, rec in enumerate(subset):
            img, gt  = rec["img"], rec["gt"]
            pred, prob = rec["pred"], rec["prob"]
            dice     = rec["dice"]
            fid      = rec["fid"]

            row_panels = [
                img,
                _overlay(img, gt,   (80, 220, 80)),
                _overlay(img, pred, (80, 160, 255)),
                None,
            ]
            for col, panel in enumerate(row_panels):
                ax = axes[row, col]
                if col == 3:
                    ax.imshow(prob, cmap=_MEDICAL_CMAP, vmin=0, vmax=1)
                    ax.contour(prob, levels=[0.5],
                               colors=["white"], linewidths=0.8, linestyles="--")
                    if gt.sum() > 0:
                        ax.contour(gt.astype(np.float32), levels=[0.5],
                                   colors=["#FFD700"], linewidths=1.0)
                else:
                    ax.imshow(panel)
                ax.axis("off")

            axes[row, 0].set_ylabel(
                f"{fid}\nDice={dice:.4f}", fontsize=7, color="white",
                rotation=0, labelpad=68, va="center",
            )

        fig.suptitle(title, fontsize=12, color="white",
                     fontweight="bold", y=0.985)
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ✓ {out_path}")

    valid = [r for r in records if not math.isnan(r["dice"])]
    srt   = sorted(valid, key=lambda x: x["dice"])

    _draw(srt[:n_worst],          f"Worst {min(n_worst,len(srt))} Frames (Lowest Dice)",
          os.path.join(save_dir, "worst_cases.png"))
    _draw(list(reversed(srt))[:n_best], f"Best {min(n_best,len(srt))} Frames (Highest Dice)",
          os.path.join(save_dir, "best_cases.png"))


def save_dice_histogram(
    dice_list: List[float],
    save_dir:  str,
    dpi:       int = 130,
) -> None:

    if not _HAS_MPL:
        return
    valid = [d for d in dice_list if not math.isnan(d)]
    if not valid:
        return

    fig, ax = plt.subplots(figsize=(9, 4), dpi=dpi, facecolor="#12121f")
    ax.set_facecolor("#1e1e2e")

    n, bins, patches = ax.hist(valid, bins=30, range=(0, 1),
                               edgecolor="#1a1a2e", linewidth=0.6, alpha=0.90)
    for patch, bc in zip(patches, 0.5*(bins[:-1]+bins[1:])):
        patch.set_facecolor((max(0,1-2*bc), min(1,2*bc), 0.25, 0.90))

    mean_d, std_d = float(np.mean(valid)), float(np.std(valid))

    ax.axvline(mean_d, color="#FFD700", lw=2.0, ls="--",
               label=f"Mean: {mean_d:.4f}")
    ax.axvline(mean_d - std_d, color="#aaa", lw=1.0, ls=":", alpha=0.7)
    ax.axvline(mean_d + std_d, color="#aaa", lw=1.0, ls=":",
               alpha=0.7, label=f"±1σ ({std_d:.4f})")

    ax.set_xlabel("Dice Score", color="white", fontsize=11)
    ax.set_ylabel("Number of Frames", color="white", fontsize=11)
    ax.set_title("Validation Set Dice Distribution", color="white",
                 fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color("#555")
    ax.legend(fontsize=9, facecolor="#2a2a3e",
              edgecolor="gray", labelcolor="white")

    out = os.path.join(save_dir, "dice_histogram.png")
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {out}")


def _pixel_stats(p: np.ndarray, t: np.ndarray) -> Tuple[int,int,int,int]:
    H, W = p.shape
    tp = int((p & t).sum())
    fp = int((p & ~t.astype(bool)).sum())
    fn = int((~p.astype(bool) & t).sum())
    tn = (H * W) - (tp + fp + fn)
    return tp, fp, fn, tn


def _hd95(p: np.ndarray, t: np.ndarray,
          sp: Tuple[float,float]=(1.,1.)) -> float:
    if not _HAS_SCIPY:
        return float("nan")
    pp = np.argwhere(p==1).astype(np.float32) * np.array(sp)
    tp = np.argwhere(t==1).astype(np.float32) * np.array(sp)
    if len(pp)==0 and len(tp)==0: return float("nan")
    if len(pp)==0 or  len(tp)==0: return float("inf")
    d1 = cKDTree(tp).query(pp)[0]
    d2 = cKDTree(pp).query(tp)[0]
    return float(max(np.percentile(d1,95), np.percentile(d2,95)))


def count_parameters(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def measure_fps(model, device, image_size=(512,512),
                warmup=20, repeats=100) -> float:
    model.eval()
    H, W  = image_size
    dummy = torch.randn(1, 3, H, W, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return repeats / (time.perf_counter() - t0)


@torch.no_grad()
def evaluate(
    checkpoint:  str,
    data_root:   Optional[str]          = None,
    use_lcc:     bool                   = True,
    threshold:   float                  = 0.5,
    hd_spacing:  Tuple[float,float]     = (1.0, 1.0),
    device_str:  Optional[str]          = None,
    vis_dir:     Optional[str]          = None,
    vis_n:       int                    = 30,
    vis_all:     bool                   = False,
    vis_skip_tn: bool                   = True,
) -> Dict:


    device = torch.device(
        device_str if device_str else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    do_vis = vis_dir is not None and _HAS_MPL
    print(f"\n  Device: {device}  LCC: {'ON' if use_lcc else 'OFF'}  "
          f"Visuals: {'→ '+vis_dir if do_vis else 'OFF'}")


    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = CAFASegNet(
        in_channels      = 3,
        encoder_channels = cfg.model.encoder_channels,
        decoder_channels = cfg.model.decoder_channels,
        num_classes      = cfg.data.num_classes,
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()
    print(f"  Ckpt: {checkpoint}"
          + (f"  (epoch {ckpt['epoch']+1}, best_dice={ckpt.get('best_dice','?')})"
             if "epoch" in ckpt else ""))

    params_m = count_parameters(model)
    print("  Measuring FPS...")
    fps = measure_fps(model, device, image_size=cfg.data.image_size)


    if data_root:
        cfg.data.dataset_root = data_root
    _, val_loader = build_labeled_loaders(cfg.data, cfg.train)
    print(f"  Val Set: {len(val_loader.dataset)} frames\n")


    if do_vis:
        dir_comp = os.path.join(vis_dir, "comparisons")
        dir_heat = os.path.join(vis_dir, "heatmaps")
        dir_summ = os.path.join(vis_dir, "summary")
        dir_ovly = os.path.join(vis_dir, "direct_overlay")
        for d in [dir_comp, dir_heat, dir_summ, dir_ovly]:
            os.makedirs(d, exist_ok=True)
        print(f"  Visual Output → {vis_dir}/\n"
              f"    ├─ comparisons/      5-panel comparisons\n"
              f"    ├─ heatmaps/         confidence heatmaps\n"
              f"    ├─ summary/          summary grids + histogram\n"
              f"    └─ direct_overlay/   Direct overlap comparison\n")


    try:
        from torch.amp import autocast as _ac
        amp_ctx = lambda: _ac(device_type=device.type,
                              enabled=(device.type=="cuda"))
    except ImportError:
        from torch.cuda.amp import autocast as _ac
        amp_ctx = lambda: _ac(enabled=(device.type=="cuda"))


    dice_list: List[float] = []
    iou_list:  List[float] = []
    hd95_list: List[float] = []
    tp_all = fp_all = fn_all = tn_all = 0
    n_total = n_tn = n_fp_skip = n_fn_inf = n_hd_skip = 0
    n_vis   = 0
    vis_recs: List[Dict] = []

    for bi, batch in enumerate(val_loader):
        imgs    = batch["image"].to(device)
        targets = batch["mask"].to(device)
        fnames  = batch.get("filename", [None]*imgs.shape[0])

        with amp_ctx():
            out = model(imgs)
        prob_raw = out["pred"]

        pred_lcc = (
            keep_largest_component_tensor(prob_raw, threshold=threshold)
            if use_lcc else (prob_raw > threshold).float()
        )

        for b in range(imgs.shape[0]):
            n_total += 1
            img_rgb  = _denorm(imgs[b])
            prob_np  = prob_raw[b, 0].cpu().float().numpy()
            p_np     = (pred_lcc[b, 0].cpu().float().numpy() > 0.5).astype(np.uint8)
            t_np     = (targets[b, 0].cpu().float().numpy() > 0.5).astype(np.uint8)
            fname    = fnames[b]
            fid      = Path(str(fname)).stem if fname else f"frame_{n_total:04d}"
            sum_p, sum_t = p_np.sum(), t_np.sum()


            if sum_p == 0 and sum_t == 0:
                n_tn += 1
                dice = 1.0;  iou = 1.0
                dice_list.append(dice); iou_list.append(iou)
                tp, fp, fn, tn = _pixel_stats(p_np, t_np)
                tn_all += tn
                if do_vis and not vis_skip_tn:
                    vis_recs.append(dict(img=img_rgb,gt=t_np,pred=p_np,
                                         prob=prob_np,dice=dice,fid=fid))
                continue

            if sum_t == 0 and sum_p > 0:
                n_fp_skip += 1
                tp, fp, fn, tn = _pixel_stats(p_np, t_np)
                fp_all += fp; tn_all += tn
                continue

            hd_val = float('nan')
            sm = 1e-6

            if sum_p == 0:
                dice = 0.0;  iou = 0.0
                dice_list.append(dice); iou_list.append(iou)
                n_fn_inf += 1; n_hd_skip += 1
                tp, fp, fn, tn = _pixel_stats(p_np, t_np)
                fn_all += fn; tn_all += tn
            else:
                inter = float((p_np & t_np).sum())
                union = float((p_np | t_np).sum())
                dice  = (2*inter+sm) / (float(sum_p)+float(sum_t)+sm)
                iou   = (inter+sm)   / (union+sm)
                dice_list.append(dice); iou_list.append(iou)
                tp, fp, fn, tn = _pixel_stats(p_np, t_np)
                tp_all += tp; fp_all += fp; fn_all += fn; tn_all += tn

                hd_val = _hd95(p_np, t_np, hd_spacing)
                if math.isinf(hd_val) or math.isnan(hd_val):
                    n_hd_skip += 1
                else:
                    hd95_list.append(hd_val)


            if do_vis:
                vis_recs.append(dict(img=img_rgb, gt=t_np, pred=p_np,
                                     prob=prob_np, dice=dice, fid=fid))


                if vis_all or n_vis < vis_n:
                    n_vis += 1
                    save_comparison(
                        img_rgb, t_np, p_np, prob_np,
                        dice=dice, iou=iou,
                        path=os.path.join(dir_comp, f"{fid}_comparison.png"),
                        fid=fid,
                    )
                    save_heatmap(
                        img_rgb, prob_np, t_np,
                        dice=dice,
                        path=os.path.join(dir_heat, f"{fid}_heatmap.png"),
                        fid=fid,
                    )

                    save_direct_overlay(
                        img_rgb, t_np, p_np,
                        dice=dice, hd95=hd_val,
                        path=os.path.join(dir_ovly, f"{fid}_direct_overlay.png"),
                        fid=fid,
                    )


        step = max(1, len(val_loader)//5)
        if (bi+1) % step == 0:
            cd = float(np.mean(dice_list)) if dice_list else 0.0
            print(f"    [{bi+1:4d}/{len(val_loader)}]  mDSC={cd:.4f}  "
                  f"Frames={n_total}" + (f"  SavedVis={n_vis}" if do_vis else ""))


    if do_vis and vis_recs:
        print(f"\n  Generating Summary Plots...")
        save_summary_grid(vis_recs, dir_summ, n_worst=12, n_best=12)
        save_dice_histogram(dice_list, dir_summ)
        print(f"  Single Plots: {n_vis} sets (comparisons/ + heatmaps/ + direct_overlay/)")


    def _ms(lst):
        if not lst: return 0., 0.
        a = np.array(lst, np.float64)
        return float(a.mean()), float(a.std())

    mdsc,  mdsc_s = _ms(dice_list)
    miou,  miou_s = _ms(iou_list)
    hd95,  hd95_s = _ms(hd95_list)
    eps = 1e-8
    prec = tp_all / (tp_all + fp_all + eps)
    rec  = tp_all / (tp_all + fn_all + eps)
    pix_acc = (tp_all + tn_all) / (tp_all + fp_all + fn_all + tn_all + eps)

    return dict(
        mDSC=mdsc, mDSC_std=mdsc_s,
        mIoU=miou, mIoU_std=miou_s,
        HD95=hd95, HD95_std=hd95_s,
        Params_M=params_m, FPS=fps,
        Precision=prec, Recall=rec, Pixel_Acc=pix_acc,
        n_total=n_total, n_valid=len(dice_list),
        n_tn=n_tn, n_fp_skip=n_fp_skip,
        n_fn_inf=n_fn_inf, n_hd_valid=len(hd95_list),
        n_hd_skip=n_hd_skip,
        TP=tp_all, FP=fp_all, FN=fn_all, TN=tn_all,
        n_vis_saved=n_vis,
    )


def print_results(r: Dict, checkpoint: str) -> None:
    sep  = "═" * 62
    sep2 = "─" * 62
    print(f"\n  {sep}")
    print(f"  {'CAFASeg-Net Evaluation Results':^58}")
    print(f"  {sep}")
    print(f"  Checkpt: {Path(checkpoint).name}")
    print(f"  {sep2}")

    rows = [
        ["mDSC  ↑",     f"{r['mDSC']:.4f}",     f"±{r['mDSC_std']:.4f}"],
        ["mIoU  ↑",     f"{r['mIoU']:.4f}",     f"±{r['mIoU_std']:.4f}"],
        ["HD95  ↓(px)", f"{r['HD95']:.2f}",      f"±{r['HD95_std']:.2f}"],
        ["Precision ↑", f"{r['Precision']:.4f}", "—"],
        ["Recall    ↑", f"{r['Recall']:.4f}",    "—"],
        ["Pixel Acc ↑", f"{r['Pixel_Acc']:.4f}", "—"],
        ["Params(M)",   f"{r['Params_M']:.2f}M",  "—"],
        ["FPS       ↑", f"{r['FPS']:.1f}",        "—"],
    ]
    if _HAS_TABULATE:
        for ln in tabulate(rows, headers=["Metric","Mean","Std"],
                           tablefmt="rounded_outline",
                           colalign=("left","right","right")).split("\n"):
            print(f"  {ln}")
    else:
        print(f"  {'Metric':<16}  {'Mean':>10}  {'Std':>10}")
        print(f"  {sep2}")
        for row in rows:
            print(f"  {row[0]:<16}  {row[1]:>10}  {row[2]:>10}")

    print(f"\n  {sep2}  Frame Statistics")
    frows = [
        ["Total Frames",               r["n_total"]],
        ["Valid DSC/IoU Frames",      r["n_valid"]],
        ["  └ True Negatives (TN)",   r["n_tn"]],
        ["  └ GT contains Objects",         r["n_valid"] - r["n_tn"]],
        ["GT=0,Pred>0 (Analytical TN)",   r["n_fp_skip"]],
        ["Missed foreground (GT>0, Pred=0)",r["n_fn_inf"]],
        ["Valid HD95 Frames",         r["n_hd_valid"]],
        ["Pixel TP/FP/FN/TN",
         f"{r['TP']:,}/{r['FP']:,}/{r['FN']:,}/{r['TN']:,}"],
    ]
    if _HAS_TABULATE:
        for ln in tabulate(frows, tablefmt="simple",
                           colalign=("left","right")).split("\n"):
            print(f"  {ln}")
    else:
        for fr in frows:
            print(f"  {fr[0]:<28}  {fr[1]}")
    print(f"  {sep}\n")


def main():
    pa = argparse.ArgumentParser(
        description="CAFASeg-Net evaluation script"
    )
    pa.add_argument("--checkpoint", default=cfg.eval.checkpoint)
    pa.add_argument("--data_root",  default=None)
    pa.add_argument("--no_lcc",     action="store_true")
    pa.add_argument("--threshold",  type=float, default=0.5)
    pa.add_argument("--device",     default=None)
    pa.add_argument("--spacing",    type=float, nargs=2, default=[1.0,1.0],
                    metavar=("SY","SX"))
    pa.add_argument("--vis_dir",    default="eval_vis",
                    help="Visualizations output dir (default: eval_vis/)")
    pa.add_argument("--vis_n",      type=int, default=30,
                    help="Max frames to save per visualization style (default: 30)")
    pa.add_argument("--vis_all",    action="store_true",
                    help="Save all frames single plots (Overrides vis_n)")
    pa.add_argument("--no_vis",     action="store_true",
                    help="Disable visualizations (Metrics only)")
    pa.add_argument("--show_tn",    action="store_true",
                    help="Include true negative frames in single visualization styles")
    args = pa.parse_args()

    require_runtime_assets()

    print(f"\n{'═'*62}")
    print(f"{'  CAFASeg-Net Evaluation':^62}")
    print(f"{'═'*62}")

    r = evaluate(
        checkpoint  = args.checkpoint,
        data_root   = args.data_root,
        use_lcc     = not args.no_lcc,
        threshold   = args.threshold,
        hd_spacing  = tuple(args.spacing),
        device_str  = args.device,
        vis_dir     = None if args.no_vis else args.vis_dir,
        vis_n       = args.vis_n,
        vis_all     = args.vis_all,
        vis_skip_tn = not args.show_tn,
    )

    print_results(r, args.checkpoint)

    if r["n_vis_saved"] > 0:
        vd = args.vis_dir
        print(f"\n  Visualizations saved to → {vd}/")
        print(f"    comparisons/      {r['n_vis_saved']} multi-panel plots")
        print(f"    heatmaps/         {r['n_vis_saved']} heatmap plots")
        print(f"    direct_overlay/   {r['n_vis_saved']} overlap comparisons")
        print(f"    summary/          worst_cases.png, best_cases.png, "
              f"dice_histogram.png")
    print()


if __name__ == "__main__":
    main()
