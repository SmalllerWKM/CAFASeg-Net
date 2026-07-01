import os
from dataclasses import dataclass, field
from typing import Tuple, List, Optional


def require_runtime_assets() -> None:
    raise RuntimeError("Runtime assets are not included in this repository package.")


@dataclass
class DataConfig:
    dataset_root:   str  = "/root/autodl-tmp/PELD/labeled_images/"
    train_img_dir:  str  = "train/images"
    train_msk_dir:  str  = "train/masks"
    val_img_dir:    str  = "val/images"
    val_msk_dir:    str  = "val/masks"


    unlabeled_dirs: List[str] = field(default_factory=lambda: [
        "unlabeled_videos/fu28",
        "unlabeled_videos/guo20",
    ])
    image_size:      Tuple[int, int] = (256, 256)
    mask_threshold: float = 0.35
    num_classes:    int   = 1


@dataclass
class ModelConfig:


    encoder_channels: Tuple[int, ...] = (64, 128, 256, 512)


    decoder_channels: Tuple[int, ...] = (128, 64, 32, 16)

    in_channels:      int   = 3


    berd_gamma_init:  float = 0.1


    head_channels:    int   = 16


@dataclass
class TrainConfig:

    seed:               int   = 42
    num_epochs:         int   = 100
    batch_size:         int   = 16
    unlabeled_ratio:    float = 1.0
    num_workers:        int   = 4
    pin_memory:         bool  = True
    device:             str   = "cuda"
    amp:                bool  = True


    lr:                float = 5e-4
    weight_decay:      float = 1e-4
    grad_clip:         float = 3.0


    backbone_lr_scale: float = 0.1


    lr_min:            float = 1e-6
    warmup_epochs:     int   = 20


    tau_init:          float = 2.0
    tau_min:           float = 0.1
    tau_anneal_epochs: int   = 30


    lambda_bce:        float = 0.25
    lambda_tversky:    float = 1.5
    lambda_deep_sup3:  float = 0.3
    lambda_deep_sup2:  float = 0.15


    lambda_cc_dice:     float = 0.3
    cc_dice_bbox_pad:   int   = 16
    lcc_min_area_ratio: float = 0.001


    focal_alpha:       float = 0.75
    focal_gamma:       float = 2.0


    tversky_alpha:     float = 0.4
    tversky_beta:      float = 0.6


    beta_range: Tuple[float, float] = (0.3, 1.5)
    atmospheric_colors: List[List[float]] = field(default_factory=lambda: [
        [0.8, 0.1, 0.1],
        [0.7, 0.6, 0.2],
        [0.9, 0.8, 0.7],
    ])


    save_dir:    str  = "checkpoints"
    log_dir:     str  = "logs"
    save_every:  int  = 10
    val_every:   int  = 1
    resume:      Optional[str] = None


@dataclass
class EvalConfig:
    checkpoint:   str   = "checkpoints/best_model.pth"
    output_dir:   str   = "predictions"
    tta:          bool  = False
    hd95_spacing: Tuple[float, float] = (1.0, 1.0)


@dataclass
class Config:
    data:  DataConfig  = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval:  EvalConfig  = field(default_factory=EvalConfig)

    def __post_init__(self):
        os.makedirs(self.train.save_dir, exist_ok=True)
        os.makedirs(self.train.log_dir,  exist_ok=True)
        os.makedirs(self.eval.output_dir, exist_ok=True)

cfg = Config()
