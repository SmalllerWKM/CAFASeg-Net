import argparse
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models  import CAFASegNet
from metrics import compute_dice, compute_hd95
from config  import cfg, require_runtime_assets


def build_infer_transform(image_size: Tuple[int, int] = (512, 512)) -> A.Compose:
    H, W = image_size
    return A.Compose([
        A.Resize(H, W),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def load_image(path: str) -> np.ndarray:

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess(
    image: np.ndarray,
    transform: A.Compose,
    device: torch.device,
) -> Tuple[Tensor, Tuple[int, int]]:


    h_orig, w_orig = image.shape[:2]
    result = transform(image=image)
    tensor = result["image"].unsqueeze(0).to(device)
    return tensor, (h_orig, w_orig)


def filter_small_components(
    mask:           np.ndarray,
    min_area_ratio: float = 0.001,
) -> np.ndarray:


    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return mask

    H, W = binary.shape
    total_pixels = H * W
    min_area = max(1, int(total_pixels * min_area_ratio))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if num_labels <= 2:
        return mask


    areas      = stats[1:, cv2.CC_STAT_AREA]
    keep_ids   = [i+1 for i, a in enumerate(areas) if a >= min_area]

    if not keep_ids:

        keep_ids = [int(np.argmax(areas)) + 1]

    max_val = mask.max()
    out = np.zeros_like(binary, dtype=np.uint8)
    for lid in keep_ids:
        out[labels == lid] = max_val

    return out


def keep_largest_component(
    mask:           np.ndarray,
    min_area_ratio: float = 0.001,
) -> np.ndarray:


    return filter_small_components(mask, min_area_ratio=min_area_ratio)


def keep_largest_component_tensor(
    pred_tensor:    "Tensor",
    threshold:      float = 0.5,
    min_area_ratio: float = 0.001,
) -> "Tensor":


    import torch
    device = pred_tensor.device
    B = pred_tensor.shape[0]
    results = []

    for b in range(B):
        prob_np    = pred_tensor[b, 0].detach().cpu().float().numpy()
        binary_255 = (prob_np > threshold).astype(np.uint8) * 255

        if binary_255.any():
            cleaned_255 = filter_small_components(binary_255, min_area_ratio)
        else:
            cleaned_255 = binary_255

        cleaned_01 = torch.from_numpy(
            (cleaned_255 > 0).astype(np.float32)
        )
        results.append(cleaned_01)

    return torch.stack(results, dim=0).unsqueeze(1).to(device)

def postprocess(
    pred:          Tensor,
    original_size: Tuple[int, int],
    threshold:     float = 0.5,
    keep_largest:  bool  = True,
) -> np.ndarray:


    H_orig, W_orig = original_size
    pred_resized = F.interpolate(
        pred, size=(H_orig, W_orig), mode="bilinear", align_corners=True
    )
    mask_np = (pred_resized.squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255

    if keep_largest and mask_np.any():
        mask_np = keep_largest_component(mask_np)

    return mask_np


def overlay_mask(
    image:    np.ndarray,
    mask:     np.ndarray,
    color:    Tuple[int, int, int] = (0, 255, 100),
    alpha:    float = 0.45,
) -> np.ndarray:


    overlay = image.copy()
    color_layer = np.zeros_like(image)


    color_layer[mask > 0] = color
    overlay = cv2.addWeighted(overlay, 1 - alpha, color_layer, alpha, 0)


    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)

    return overlay


class CAFASegNetInferencer:


    def __init__(
        self,
        checkpoint:   str,
        image_size:   Tuple[int, int] = (512, 512),
        threshold:    float = 0.5,
        device:       Optional[str] = None,
        keep_largest: bool = True,
    ) -> None:
        self.threshold    = threshold
        self.image_size   = image_size
        self.keep_largest = keep_largest


        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)


        self.model = CAFASegNet(
            encoder_channels = cfg.model.encoder_channels,
            decoder_channels = cfg.model.decoder_channels,
            num_classes      = cfg.data.num_classes,
        ).to(self.device)

        self._load_weights(checkpoint)
        self.model.eval()


        self.transform = build_infer_transform(image_size)

    def _load_weights(self, checkpoint: str) -> None:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        state = torch.load(checkpoint, map_location=self.device)

        sd = state.get("model", state)
        self.model.load_state_dict(sd)
        print(f"Loaded weights: {checkpoint}")

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:


        tensor, orig_size = preprocess(image, self.transform, self.device)


        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
            out = self.model(tensor)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        mask    = postprocess(out["pred"], orig_size, self.threshold,
                              keep_largest=self.keep_largest)
        overlay = overlay_mask(image, mask)

        return mask, overlay, latency_ms


def infer_single_image(
    inferencer: CAFASegNetInferencer,
    image_path: str,
    output_dir: str,
    gt_path:    Optional[str] = None,
) -> None:

    image = load_image(image_path)
    mask, overlay, latency = inferencer.predict(image)

    stem = Path(image_path).stem
    os.makedirs(output_dir, exist_ok=True)


    cv2.imwrite(
        os.path.join(output_dir, f"{stem}_mask.png"),
        mask
    )

    cv2.imwrite(
        os.path.join(output_dir, f"{stem}_overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    print(f"[Image] {Path(image_path).name}  latency: {latency:.1f}ms  "
          f"FPS: {1000/latency:.1f}")


    if gt_path and os.path.exists(gt_path):
        gt_raw = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_bin = (gt_raw.astype(np.float32) / 255.0 > 0.5).astype(np.uint8)
        pred_bin = (mask > 0).astype(np.uint8)
        dice = compute_dice(
            torch.tensor(pred_bin).float().unsqueeze(0).unsqueeze(0),
            torch.tensor(gt_bin).float().unsqueeze(0).unsqueeze(0),
        )
        hd95 = compute_hd95(
            torch.tensor(pred_bin).float().unsqueeze(0).unsqueeze(0),
            torch.tensor(gt_bin).float().unsqueeze(0).unsqueeze(0),
        )
        print(f"       Dice={dice:.4f}  HD95={hd95:.2f}mm")


def infer_directory(
    inferencer: CAFASegNetInferencer,
    input_dir:  str,
    output_dir: str,
    gt_dir:     Optional[str] = None,
) -> None:

    input_path = Path(input_dir)
    img_paths  = sorted(list(input_path.glob("*.jpg")) +
                        list(input_path.glob("*.JPG")) +
                        list(input_path.glob("*.png")))

    if not img_paths:
        print(f"No images found in directory: {input_dir}")
        return

    latencies = []
    dice_list = []
    os.makedirs(output_dir, exist_ok=True)

    for i, img_path in enumerate(img_paths):
        image = load_image(str(img_path))
        mask, overlay, latency = inferencer.predict(image)
        latencies.append(latency)

        stem = img_path.stem
        cv2.imwrite(os.path.join(output_dir, f"{stem}_mask.png"), mask)
        cv2.imwrite(
            os.path.join(output_dir, f"{stem}_overlay.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )


        dice_str = ""
        if gt_dir:
            gt_path = Path(gt_dir) / img_path.name
            if gt_path.exists():
                gt_raw = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                gt_bin = (gt_raw.astype(np.float32) / 255.0 > 0.5).astype(np.uint8)
                pred_bin = (mask > 0).astype(np.uint8)
                dice = compute_dice(
                    torch.tensor(pred_bin).float().unsqueeze(0).unsqueeze(0),
                    torch.tensor(gt_bin).float().unsqueeze(0).unsqueeze(0),
                )
                dice_list.append(dice)
                dice_str = f"  Dice={dice:.4f}"

        if (i + 1) % 50 == 0 or i == 0:
            print(f"[{i+1:4d}/{len(img_paths)}] {img_path.name}  "
                  f"{latency:.1f}ms{dice_str}")

    mean_latency = np.mean(latencies)
    print(f"\n--- Batch inference summary --------------------------------")
    print(f"  Total samples: {len(img_paths)}")
    print(f"  Mean latency:  {mean_latency:.1f} ms")
    print(f"  Mean FPS:      {1000/mean_latency:.1f}")
    if dice_list:
        print(f"  Mean Dice:     {np.mean(dice_list):.4f} ± {np.std(dice_list):.4f}")
    print(f"  Results saved to: {output_dir}")


def infer_video(
    inferencer:  CAFASegNetInferencer,
    video_path:  str,
    output_path: str,
    fps_cap:     Optional[float] = None,
) -> None:

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W_in   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_in   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps_out = min(fps_in, fps_cap) if fps_cap else fps_in
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(output_path, fourcc, fps_out, (W_in, H_in))

    latencies = []
    frame_idx = 0

    print(f"Video inference: {Path(video_path).name}  "
          f"({W_in}x{H_in} @ {fps_in:.1f}fps, {n_frames} frames)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask, overlay, latency = inferencer.predict(frame_rgb)
        latencies.append(latency)

        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        writer.write(overlay_bgr)
        frame_idx += 1

        if frame_idx % 100 == 0:
            mean_lat = np.mean(latencies[-100:])
            print(f"  Frame {frame_idx:5d}/{n_frames}  mean latency: {mean_lat:.1f}ms  "
                  f"FPS: {1000/mean_lat:.1f}")

    cap.release()
    writer.release()

    mean_latency = np.mean(latencies)
    print(f"\n--- Video inference summary --------------------------------")
    print(f"  Processed frames: {frame_idx}")
    print(f"  Mean latency:  {mean_latency:.1f} ms")
    print(f"  Mean FPS:      {1000/mean_latency:.1f}")
    print(f"  Output video:      {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAFASegNet inference")
    parser.add_argument("--mode",       type=str, required=True,
                        choices=["image", "dir", "video"],
                        help="Inference mode: image/dir/video")
    parser.add_argument("--input",      type=str, required=True,
                        help="Input path (image, directory, or video)")
    parser.add_argument("--checkpoint", type=str,
                        default=cfg.eval.checkpoint,
                        help="Model checkpoint path")
    parser.add_argument("--output",     type=str,
                        default=cfg.eval.output_dir,
                        help="Output directory or file path")
    parser.add_argument("--gt",         type=str, default=None,
                        help="Ground-truth mask path (optional, for Dice)")
    parser.add_argument("--threshold",  type=float, default=0.5,
                        help="Binarization threshold")
    parser.add_argument("--device",     type=str, default=None,
                        help="Inference device: cuda/cpu")
    args = parser.parse_args()

    require_runtime_assets()

    inferencer = CAFASegNetInferencer(
        checkpoint = args.checkpoint,
        image_size = cfg.data.image_size,
        threshold  = args.threshold,
        device     = args.device,
    )

    if args.mode == "image":
        infer_single_image(
            inferencer, args.input, args.output, gt_path=args.gt
        )
    elif args.mode == "dir":
        infer_directory(
            inferencer, args.input, args.output, gt_dir=args.gt
        )
    elif args.mode == "video":
        output_file = args.output
        if os.path.isdir(output_file):
            output_file = os.path.join(output_file, "output.mp4")
        infer_video(inferencer, args.input, output_file)
