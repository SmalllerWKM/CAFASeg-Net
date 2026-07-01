import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGE_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

def collect_images(directory: Path) -> List[Path]:
    paths: List[Path] = []
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS:
            paths.append(p)
    return paths

def find_mask_for_image(img_path: Path, mask_dir: Path) -> Optional[Path]:
    exact = mask_dir / img_path.name
    if exact.exists():
        return exact
    stem = img_path.stem
    for ext in IMAGE_EXTS:
        for candidate_ext in (ext, ext.upper(), ext.capitalize()):
            candidate = mask_dir / f"{stem}{candidate_ext}"
            if candidate.exists():
                return candidate
    return None

def safe_imread_rgb(path: Path) -> Optional[np.ndarray]:
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def safe_imread_gray(path: Path) -> Optional[np.ndarray]:
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

def load_mask(mask_raw: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if mask_raw.max() <= 1:
        return (mask_raw > 0).astype(np.uint8)
    return (mask_raw.astype(np.float32) / 255.0 > threshold).astype(np.uint8)


def _smooth_noise_field(h: int, w: int, scale: int = 32) -> np.ndarray:
    lh, lw = max(1, h // scale), max(1, w // scale)
    noise_low = np.random.rand(lh, lw).astype(np.float32)
    noise = cv2.resize(noise_low, (w, h), interpolation=cv2.INTER_LINEAR)
    lo, hi = noise.min(), noise.max()
    return (noise - lo) / (hi - lo + 1e-8)

def mcCartney_degrade(
    img: np.ndarray,
    beta_range: Tuple[float, float] = (0.3, 1.5),
    atm_colors: Optional[List[List[float]]] = None,
) -> np.ndarray:
    if atm_colors is None:
        atm_colors = [[0.8, 0.1, 0.1], [0.7, 0.6, 0.2], [0.9, 0.8, 0.7]]
    H, W = img.shape[:2]
    beta = random.uniform(*beta_range)
    d    = _smooth_noise_field(H, W) * 0.8 + 0.2
    t    = np.exp(-beta * d)[..., np.newaxis]
    A    = np.array(random.choice(atm_colors), dtype=np.float32).reshape(1, 1, 3)
    return np.clip(img * t + A * (1.0 - t), 0.0, 1.0)


def foreground_aware_crop(
    image:      np.ndarray,
    mask:       np.ndarray,
    crop_size:  Tuple[int, int],
    fg_prob:    float = 0.7,
    min_fg_px:  int   = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    H, W = image.shape[:2]
    cH, cW = crop_size

    cH = min(cH, H)
    cW = min(cW, W)

    fg_coords = np.argwhere(mask == 1)

    use_fg_center = (
        len(fg_coords) >= min_fg_px
        and random.random() < fg_prob
    )

    if use_fg_center:
        cy, cx = fg_coords[random.randint(0, len(fg_coords) - 1)]
        jitter_y = random.randint(-cH // 4, cH // 4)
        jitter_x = random.randint(-cW // 4, cW // 4)
        cy = int(cy) + jitter_y
        cx = int(cx) + jitter_x
    else:
        cy = random.randint(cH // 2, H - cH // 2)
        cx = random.randint(cW // 2, W - cW // 2)

    y1 = max(0, cy - cH // 2)
    x1 = max(0, cx - cW // 2)
    y1 = min(y1, H - cH)
    x1 = min(x1, W - cW)
    y2 = y1 + cH
    x2 = x1 + cW

    return image[y1:y2, x1:x2], mask[y1:y2, x1:x2]


def copy_paste_foreground(
    image:      np.ndarray,
    mask:       np.ndarray,
    src_image:  np.ndarray,
    src_mask:   np.ndarray,
    prob:       float = 0.3,
    min_fg_px:  int   = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() > prob:
        return image, mask

    fg_pixels = np.sum(src_mask == 1)
    if fg_pixels < min_fg_px:
        return image, mask

    num_labels, labels = cv2.connectedComponents(src_mask)
    if num_labels <= 1:
        return image, mask

    chosen_label = random.randint(1, num_labels - 1)
    component_mask = (labels == chosen_label).astype(np.uint8)

    if component_mask.sum() < min_fg_px:
        return image, mask

    H, W = image.shape[:2]

    ys, xs = np.where(component_mask == 1)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    patch_h = y_max - y_min + 1
    patch_w = x_max - x_min + 1

    if patch_h > H or patch_w > W:
        return image, mask

    dst_y = random.randint(0, H - patch_h)
    dst_x = random.randint(0, W - patch_w)

    src_region = src_image[y_min:y_max+1, x_min:x_max+1]
    fg_region  = component_mask[y_min:y_max+1, x_min:x_max+1]

    dst_image = image.copy()
    dst_mask  = mask.copy()

    fg_mask_3c = fg_region[:, :, np.newaxis].astype(bool)
    dst_image[dst_y:dst_y+patch_h, dst_x:dst_x+patch_w] = np.where(
        fg_mask_3c,
        src_region,
        dst_image[dst_y:dst_y+patch_h, dst_x:dst_x+patch_w]
    )
    dst_mask[dst_y:dst_y+patch_h, dst_x:dst_x+patch_w] = np.maximum(
        dst_mask[dst_y:dst_y+patch_h, dst_x:dst_x+patch_w],
        fg_region
    )

    return dst_image, dst_mask


def build_train_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Affine(
            translate_percent=0.05,
            scale=(0.9, 1.1),
            rotate=(-20, 20),
            p=0.4,
        ),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.6),
        A.CLAHE(clip_limit=3.0, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(std_range=(0.01, 0.04), p=0.25),
        A.ElasticTransform(alpha=30, sigma=5, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def build_val_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def build_weak_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class PELDLabeledDataset(Dataset):
    def __init__(
        self,
        img_dir:        str,
        mask_dir:       str,
        transform:      A.Compose,
        mask_threshold: float = 0.5,
        use_fg_crop:    bool  = True,
        fg_crop_prob:   float = 0.7,
        use_copy_paste: bool  = True,
        copy_paste_prob: float = 0.3,
    ) -> None:
        self.img_dir        = Path(img_dir)
        self.mask_dir       = Path(mask_dir)
        self.transform      = transform
        self.mask_thr       = mask_threshold
        self.use_fg_crop    = use_fg_crop
        self.fg_crop_prob   = fg_crop_prob
        self.use_copy_paste = use_copy_paste
        self.copy_paste_prob = copy_paste_prob

        all_images = collect_images(self.img_dir)
        if not all_images:
            raise FileNotFoundError(
                f"No supported image files found in the image directory\n"
                f"  directory: {img_dir}\n  supported extensions: {IMAGE_EXTS}"
            )

        self.samples: List[Tuple[Path, Path]] = []
        skipped = 0
        for img_p in all_images:
            msk_p = find_mask_for_image(img_p, self.mask_dir)
            if msk_p is None:
                warnings.warn(f"[PELDLabeledDataset] Mask not found, skipped: {img_p.name}")
                skipped += 1
                continue
            self.samples.append((img_p, msk_p))

        if not self.samples:
            raise FileNotFoundError(
                f"No image/mask pairs found\n"
                f"  img_dir:  {img_dir}\n  mask_dir: {mask_dir}"
            )
        if skipped > 0:
            warnings.warn(f"[PELDLabeledDataset] Skipped {skipped} images without masks")

        print(f"  Computing foreground ratios for {len(self.samples)} samples...")
        self.fg_ratios: List[float] = self._compute_fg_ratios()
        fg_nonzero = sum(1 for r in self.fg_ratios if r > 0)
        avg_fg = sum(self.fg_ratios) / max(len(self.fg_ratios), 1)
        print(f"  Images with foreground: {fg_nonzero}/{len(self.samples)}  "
              f"mean foreground ratio: {avg_fg:.4f} ({avg_fg*100:.2f}%)")

    def _compute_fg_ratios(self) -> List[float]:
        ratios = []
        for _, msk_p in self.samples:
            raw = safe_imread_gray(msk_p)
            if raw is None:
                ratios.append(0.0)
                continue
            m = load_mask(raw, self.mask_thr)
            ratios.append(float(m.mean()))
        return ratios

    def get_sample_weights(self) -> List[float]:


        weights = []
        for r in self.fg_ratios:
            if r > 0:
                weights.append(r + 0.1)
            else:


                weights.append(0.05)
        return weights

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        img_path, msk_path = self.samples[idx]

        image = safe_imread_rgb(img_path)
        if image is None:
            raise IOError(f"Failed to read image: {img_path}")
        mask_raw = safe_imread_gray(msk_path)
        if mask_raw is None:
            raise IOError(f"Failed to read mask: {msk_path}")
        mask = load_mask(mask_raw, self.mask_thr)

        if self.use_copy_paste:
            src_idx = random.randint(0, len(self.samples) - 1)
            if src_idx != idx:
                src_img_p, src_msk_p = self.samples[src_idx]
                src_img = safe_imread_rgb(src_img_p)
                src_raw = safe_imread_gray(src_msk_p)
                if src_img is not None and src_raw is not None:
                    src_msk = load_mask(src_raw, self.mask_thr)
                    if self.fg_ratios[src_idx] > 0.005:
                        image, mask = copy_paste_foreground(
                            image, mask, src_img, src_msk,
                            prob=self.copy_paste_prob,
                        )

        H, W = image.shape[:2]
        if self.use_fg_crop and (H > 512 or W > 512):
            image, mask = foreground_aware_crop(
                image, mask,
                crop_size=(min(H, 800), min(W, 800)),
                fg_prob=self.fg_crop_prob,
            )

        result = self.transform(image=image, mask=mask)
        return {
            "image": result["image"],
            "mask":  result["mask"].unsqueeze(0).float(),
            "path":  str(img_path),
        }


class PELDUnlabeledDataset(Dataset):
    def __init__(
        self,
        video_dirs:  List[str],
        beta_range:  Tuple[float, float] = (0.3, 1.5),
        atm_colors:  Optional[List[List[float]]] = None,
    ) -> None:
        self.beta_range = beta_range
        self.atm_colors = atm_colors

        self.frame_paths: List[Path] = []
        for d in video_dirs:
            root = Path(d)
            if not root.exists():
                warnings.warn(f"[PELDUnlabeledDataset] Directory does not exist: {d}")
                continue
            self.frame_paths.extend(collect_images(root))

        if not self.frame_paths:
            raise FileNotFoundError(
                f"No images found in the unlabeled directories\n"
                f"  directories: {video_dirs}\n  supported extensions: {IMAGE_EXTS}"
            )

        self.weak_tf  = build_weak_transforms()
        self.resize_tf = A.Compose([A.Resize(512, 512)])
        self.norm_tf  = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        frame_path = self.frame_paths[idx]
        frame = safe_imread_rgb(frame_path)

        if frame is None:
            warnings.warn(f"[PELDUnlabeledDataset] Failed to read frame; using zeros: {frame_path}")
            zero = np.zeros((512, 512, 3), dtype=np.uint8)
            z = self.norm_tf(image=zero)["image"]
            return {"x_weak": z, "x_strong": z, "path": str(frame_path)}

        x_weak = self.weak_tf(image=frame)["image"]

        resized  = self.resize_tf(image=frame)["image"]
        img_f32  = resized.astype(np.float32) / 255.0
        degraded = mcCartney_degrade(img_f32, self.beta_range, self.atm_colors)
        deg_u8   = (degraded * 255.0).clip(0, 255).astype(np.uint8)
        x_strong = self.norm_tf(image=deg_u8)["image"]

        return {
            "x_weak":   x_weak,
            "x_strong": x_strong,
            "path":     str(frame_path),
        }


def build_labeled_loaders(
    cfg_data,
    cfg_train,
    dataset_root: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    root = Path(dataset_root or cfg_data.dataset_root)

    train_ds = PELDLabeledDataset(
        img_dir         = root / cfg_data.train_img_dir,
        mask_dir        = root / cfg_data.train_msk_dir,
        transform       = build_train_transforms(),
        mask_threshold  = cfg_data.mask_threshold,
        use_fg_crop     = True,
        fg_crop_prob    = 0.7,
        use_copy_paste  = True,
        copy_paste_prob = 0.3,
    )
    val_ds = PELDLabeledDataset(
        img_dir         = root / cfg_data.val_img_dir,
        mask_dir        = root / cfg_data.val_msk_dir,
        transform       = build_val_transforms(),
        mask_threshold  = cfg_data.mask_threshold,
        use_fg_crop     = False,
        use_copy_paste  = False,
    )

    sample_weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(train_ds),
        replacement = True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg_train.batch_size,
        sampler     = sampler,
        num_workers = cfg_train.num_workers,
        pin_memory  = cfg_train.pin_memory,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg_train.batch_size,
        shuffle     = False,
        num_workers = cfg_train.num_workers,
        pin_memory  = cfg_train.pin_memory,
    )
    return train_loader, val_loader

def build_unlabeled_loader(
    cfg_data,
    cfg_train,
    dataset_root: Optional[str] = None,
) -> DataLoader:
    root = Path(dataset_root or cfg_data.dataset_root)
    dirs = [str(root / d) for d in cfg_data.unlabeled_dirs]
    unlabeled_ds = PELDUnlabeledDataset(
        video_dirs  = dirs,
        beta_range  = cfg_train.beta_range,
        atm_colors  = cfg_train.atmospheric_colors,
    )
    batch_size = max(1, int(cfg_train.batch_size * cfg_train.unlabeled_ratio))
    return DataLoader(
        unlabeled_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = cfg_train.num_workers,
        pin_memory  = cfg_train.pin_memory,
        drop_last   = True,
    )

def diagnose_dataset(img_dir: str, mask_dir: str) -> None:
    img_dir_p  = Path(img_dir)
    mask_dir_p = Path(mask_dir)
    all_images = collect_images(img_dir_p)

    fg_ratios = []
    unmatched = 0
    for img_p in all_images:
        msk_p = find_mask_for_image(img_p, mask_dir_p)
        if msk_p is None:
            unmatched += 1
            continue
        raw = safe_imread_gray(msk_p)
        if raw is None:
            continue
        m = load_mask(raw)
        fg_ratios.append(float(m.mean()))

    if not fg_ratios:
        print("No valid samples found.")
        return

    fg_ratios_arr = np.array(fg_ratios)
    n_total   = len(fg_ratios)
    n_fg      = (fg_ratios_arr > 0).sum()
    n_bg_only = n_total - n_fg

    print(f"{'='*55}")
    print(f"Dataset class-imbalance diagnostics")
    print(f"{'='*55}")
    print(f"Total samples:              {n_total}")
    print(f"Samples with foreground:    {n_fg}  ({n_fg/n_total*100:.1f}%)")
    print(f"Background-only samples:    {n_bg_only}  ({n_bg_only/n_total*100:.1f}%)")
    print(f"Images without masks:       {unmatched}")
    print(f"---")
    print(f"Mean foreground ratio:      {fg_ratios_arr.mean()*100:.3f}%")
    print(f"Median foreground ratio:    {np.median(fg_ratios_arr)*100:.3f}%")
    print(f"Max foreground ratio:       {fg_ratios_arr.max()*100:.3f}%")
    print(f"Min foreground ratio:       {fg_ratios_arr[fg_ratios_arr>0].min()*100:.3f}%"
          if n_fg > 0 else "Min foreground ratio:       N/A")
    print(f"{'='*55}")

    bins = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.01]
    labels = ["0%", "0~1%", "1~5%", "5~10%", "10~20%", "20~50%", ">50%"]
    counts = []
    for i in range(len(bins)-1):
        cnt = ((fg_ratios_arr >= bins[i]) & (fg_ratios_arr < bins[i+1])).sum()
        counts.append(int(cnt))
    print("Foreground-ratio distribution:")
    max_cnt = max(counts) if counts else 1
    for lbl, cnt in zip(labels, counts):
        bar = "█" * int(cnt / max_cnt * 30)
        print(f"  {lbl:>8}  {bar:<30}  {cnt}")
    print(f"{'='*55}")
