# CAFASeg-Net

**CAFASeg-Net** is a neural tissue segmentation framework for endoscopic videos, specifically designed for scenarios involving **extreme class imbalance** foreground pixels < 5% and **small target structures**.

---

## Table of Contents

* [Framework Overview](#framework-overview)
* [Model Architecture](#model-architecture)
* [Directory Structure](#directory-structure)
* [Environment Setup](#environment-setup)
* [Data Preparation](#data-preparation)
* [Training](#training)
* [Evaluation](#evaluation)
* [Inference](#inference)
* [Semi-supervised Extension Routes](#semi-supervised-extension-routes)
* [Configuration Parameters](#configuration-parameters)
* [FAQ](#faq)

---

## Framework Overview

```text
Raw annotated data (data/)
       ↓
  prepare_dataset.py      ← Data preprocessing: FOV cropping / Mask generation / train-val-test split
       ↓
  PELD/labeled_images/    ← Standardized dataset (512×512 PNG)
       ↓
    train.py              ← Main supervised training + semi-supervised PCCP joint training
       ↓
  checkpoints/            ← Model weights
       ↓
  evaluate.py             ← Test-set metric evaluation + visualization
  inference.py            ← Single-image / image-folder / video inference
  video_infer_V2.py       ← Advanced video inference with confidence filtering
```

---

## Model Architecture

### Backbone: DSBA-Bone Dual-Stream Boundary-Aware Backbone

```text
Input (3×512×512)
    ├── Semantic Stream
    │     └── ResNet34 ImageNet-pretrained
    │           ├── Stage1: [B, 64,  H/4,  W/4]
    │           ├── Stage2: [B, 128, H/8,  W/8]
    │           ├── Stage3: [B, 256, H/16, W/16]
    │           └── Stage4: [B, 512, H/32, W/32]
    │
    └── Boundary Stream
          └── Lightweight convolution + CrossStreamInjection
                ├── f1: [B, 64,  H/4, W/4]  ← injected into Semantic Stream Stage1
                └── f2: [B, 128, H/8, W/8]  ← injected into Semantic Stream Stage2
         ↓
    Bottleneck: ASPPLite Atrous Spatial Pyramid Pooling, four branches: r=1,6,12,GAP
         ↓
    Decoder: 4× DecoderBlock CAFM cross-scale adaptive feature fusion + bilinear upsampling
         ↓
    Refinement: BERD Sobel-gradient-modulated boundary enhancement
         ↓
    Output head 1×512×512 Sigmoid
```

### Core Modules

| Module        | Full Name                            | Function                                                                                             |
| ------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| **DSBA-Bone** | Dual-Stream Boundary-Aware Backbone  | A heterogeneous dual-stream backbone with semantic stream and boundary stream cross-stream injection |
| **ASPPLite**  | Atrous Spatial Pyramid Pooling Lite  | Multi-scale context aggregation with dilation rates 1/6/12 and global average pooling                |
| **CAFM**      | Cross-scale Adaptive Feature Merging | Channel-attention and spatial-attention gated fusion of encoder-decoder features                     |
| **BERD**      | Boundary-Enhanced Refinement Decoder | Sobel gradient detection with learnable γ modulation for boundary response enhancement               |

> Compatible alternative backbones: `backbone="segformer"` requires `transformers`, and `backbone="arfe"` uses Gumbel-Softmax dynamic routing.

---

## Directory Structure

```text
CAFASeg-Net/
├── config.py                  # Global hyperparameter configuration center
├── prepare_dataset.py         # Dataset preprocessing script
├── dataset.py                 # Dataset / DataLoader / augmentation pipeline
├── models.py                  # Model architecture definitions
├── losses.py                  # Loss functions Tversky / CCDice / PCCP
├── metrics.py                 # Evaluation metrics Dice / IoU / HD95
├── train.py                   # Main training program
├── evaluate.py                # Test-set evaluation and visualization
├── inference.py               # Single-image / folder / video inference
├── video_infer.py             # Stable video inference with ROI cropping
├── video_infer_V2.py          # Full video inference with stream inference and confidence filtering
├── route1_pseudo_label.py     # Semi-supervised route 1: pseudo-labeling
├── route3_optical_flow.py     # Semi-supervised route 3: optical-flow temporal consistency
└── requirements.txt           # Dependency list

# Automatically generated after running:
PELD/
└── labeled_images/
    ├── train/
    │   ├── images/
    │   └── masks/
    ├── val/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/

checkpoints/                   # Directory for saving model weights
logs/                          # TensorBoard training logs
predictions/                   # Prediction outputs from evaluate.py
```

---

## Environment Setup

### 1. Create a virtual environment

```bash
conda create -n cafaseg python=3.10 -y
conda activate cafaseg
```

### 2. Install PyTorch using CUDA 11.8 as an example

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

> Please select the corresponding installation command from [pytorch.org](https://pytorch.org/get-started/locally/) according to your actual CUDA version.

### 3. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify the installation

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python models.py   # Run the built-in test to output the logits shape and parameter count
```

---

## Data Preparation

### Raw Data Format

```text
data/
├── images/
│   ├── case001.png
│   └── ...
└── labels/
    ├── case001.json    # LabelMe format, polygon vertices stored in shapes[].points
    └── ...
```

### Run Preprocessing

```bash
# Basic usage with default paths and parameters
python prepare_dataset.py

# Specify paths + accelerate with multiprocessing + generate visualization checks
python prepare_dataset.py --src /root/autodl-tmp/data --dst /root/autodl-tmp/PELD --workers 8 --vis

# Customize the split ratio, for example 7:2:1
python prepare_dataset.py --split 0.7 0.2 0.1

# Automatically append the test-set path field to config.py
python prepare_dataset.py --patch_cfg
```

**Preprocessing workflow:**

1. Scan `data/labels/` and filter out empty annotations.
2. Generate binary masks from JSON polygons using `fillPoly`.
3. Automatically detect the circular endoscopic FOV region and crop it to the minimum enclosing square.
4. Resize to 512×512. Images use `INTER_AREA`, and masks use `INTER_NEAREST`.
5. Randomly split the dataset into `PELD/labeled_images/` using an 8:1:1 ratio.
6. Print a foreground-distribution statistical report for each split.

### Diagnose Dataset Class Imbalance

```python
from dataset import diagnose_dataset
diagnose_dataset("PELD/labeled_images/train/images",
                 "PELD/labeled_images/train/masks")
```

---

## Training

### Basic Training Fully Supervised

```bash
python train.py
```

### Common Arguments

```bash
# Specify the dataset root directory
python train.py --data_root /path/to/PELD

# Resume training from a checkpoint
python train.py --resume checkpoints/last_checkpoint.pth

# Modify batch size and learning rate
python train.py --batch_size 8 --lr 3e-4

# Disable AMP mixed precision for debugging
python train.py --no_amp
```

### Semi-supervised Joint Training Supervised + PCCP Unsupervised

Configure the unlabeled video directories in `config.py`:

```python
unlabeled_dirs: List[str] = field(default_factory=lambda: [
    "unlabeled_videos/patient_A",
    "unlabeled_videos/patient_B",
])
```

The training program automatically detects the unlabeled dataset and enables the PCCP physically driven cross-contrast pseudo-supervision loss.

### Training Logs

* TensorBoard: `tensorboard --logdir logs/`
* Excel training records: `logs/training_log.xlsx` requires `openpyxl`
* Real-time console output of Dice / Loss tables

### Differential Learning Rate Strategy DSBA-Bone

| Parameter Group                     | Learning Rate                         | Description                                                        |
| ----------------------------------- | ------------------------------------- | ------------------------------------------------------------------ |
| Pretrained semantic stream ResNet34 | `lr × backbone_lr_scale` default 5e-5 | Slow fine-tuning to preserve ImageNet priors                       |
| Boundary stream + decoder           | `lr` default 5e-4                     | Initialized from scratch and optimized with the full learning rate |

---

## Evaluation

```bash
# Evaluate on the validation set using the checkpoint configured in config.py
python evaluate.py

# Specify the weight file and output directory
python evaluate.py --checkpoint checkpoints/best_model.pth --vis_dir eval_vis

# Save visualization images for all frames
python evaluate.py --checkpoint best_model.pth --vis_all

# Compute metrics only without generating visualization images
python evaluate.py --checkpoint best_model.pth --no_vis
```

### Output Metrics

| Metric    | Description                                      |
| --------- | ------------------------------------------------ |
| mDSC      | Mean Dice Similarity Coefficient foreground only |
| mIoU      | Mean Intersection over Union foreground only     |
| HD95      | Mean 95% Hausdorff Distance mm                   |
| Precision | Precision = TP / TP + FP                         |
| Recall    | Recall = TP / TP + FN                            |
| Pixel_Acc | Pixel accuracy                                   |
| FPS       | Inference throughput frames per second           |

### Visualization Output Directory Structure

```text
eval_vis/
├── comparisons/       # Five-panel figures: original image | GT overlay | prediction overlay | confidence heatmap | error map
├── heatmaps/          # Independent confidence heatmaps
├── summary/           # Worst/best sample summary grids + Dice distribution histogram
└── direct_overlay/    # GT contour green vs prediction contour magenta comparison images
```

---

## Inference

### Single Image

```bash
python inference.py --mode image \
    --input path/to/image.jpg \
    --checkpoint checkpoints/best_model.pth \
    --output_dir predictions/
```

### Batch Inference on an Image Folder

```bash
python inference.py --mode dir \
    --input path/to/images/ \
    --checkpoint checkpoints/best_model.pth \
    --output_dir predictions/
```

### Video Inference

```bash
# Stable version ROI cropping acceleration + ordered output
python video_infer.py --mode video \
    --input surgery_video.mp4 \
    --checkpoint checkpoints/best_model.pth \
    --output result.mp4

# Full version with confidence filtering + real-time stream inference
python video_infer_V2.py --mode video \
    --input surgery_video.mp4 \
    --checkpoint checkpoints/best_model.pth \
    --output result.mp4

# Real-time camera stream available only in video_infer_V2.py
python video_infer_V2.py --mode stream \
    --input 0 \
    --checkpoint checkpoints/best_model.pth
```

---

## Semi-supervised Extension Routes

### Route 1: Pseudo-labeling `route1_pseudo_label.py`

Use a teacher model to generate high-confidence pseudo-labels for unlabeled data, then train with a mixture of manual labels and pseudo-labels.

```bash
# Step 1: Generate pseudo-labels only needs to be run once
python route1_pseudo_label.py --step generate \
    --checkpoint checkpoints/best_model.pth \
    --output_dir pseudo_labels \
    --conf_high 0.90 --conf_low 0.10 \
    --min_fg 0.01 --max_fg 0.30

# Step 2: Mixed training
python route1_pseudo_label.py --step train \
    --pseudo_dir pseudo_labels \
    --resume checkpoints/best_model.pth
```

| Parameter         | Default | Description                                                                  |
| ----------------- | ------- | ---------------------------------------------------------------------------- |
| `--conf_high`     | 0.90    | High-confidence foreground threshold                                         |
| `--conf_low`      | 0.10    | High-confidence background threshold                                         |
| `--min_fg`        | 0.01    | Minimum foreground ratio for keeping a frame                                 |
| `--max_fg`        | 0.30    | Maximum foreground ratio for keeping a frame, used to filter abnormal frames |
| `--pseudo_weight` | 0.50    | Pseudo-label loss weight, set below 1.0 to reflect uncertainty               |

### Route 3: Optical-flow Temporal Consistency `route3_optical_flow.py`

Use RAFT optical flow to align predictions between adjacent frames and construct a temporal consistency constraint.

```bash
# Step 1: Offline pre-extraction of optical flow recommended to save training time
python route3_optical_flow.py --step precompute_flow \
    --flow_dir flow_cache

# Step 2: Start training
python route3_optical_flow.py --step train \
    --resume checkpoints/best_model.pth \
    --flow_dir flow_cache

# Optional: online optical flow no pre-extraction required, but each step adds about 50 ms
python route3_optical_flow.py --step train \
    --resume checkpoints/best_model.pth \
    --online_flow
```

| Parameter       | Default | Description                                                                                     |
| --------------- | ------- | ----------------------------------------------------------------------------------------------- |
| `--lambda_flow` | 0.01    | Weight of the optical-flow temporal loss                                                        |
| `--flow_warmup` | 50      | Starting epoch for activating the optical-flow loss                                             |
| `--warp_conf`   | 0.80    | Optical-flow confidence threshold. Occluded regions below this value are excluded from the loss |

---

## Configuration Parameters

All hyperparameters are centralized in `config.py` and divided into four dataclasses.

### `DataConfig`

| Parameter        | Default                 | Description                                                   |
| ---------------- | ----------------------- | ------------------------------------------------------------- |
| `dataset_root`   | `/root/autodl-tmp/PELD` | Dataset root directory                                        |
| `image_size`     | `(512, 512)`            | Model input size                                              |
| `mask_threshold` | `0.35`                  | Mask binarization threshold                                   |
| `num_classes`    | `1`                     | Number of segmentation classes. Use 1 for binary segmentation |

### `ModelConfig`

| Parameter          | Default            | Description                                           |
| ------------------ | ------------------ | ----------------------------------------------------- |
| `backbone`         | `"dsbabone"`       | Backbone selection: `dsbabone` / `segformer` / `arfe` |
| `encoder_channels` | `(64,128,256,512)` | Output channels of each encoder stage                 |
| `decoder_channels` | `(128,64,32,16)`   | Output channels of each decoder stage                 |

### `TrainConfig` Key Parameters

| Parameter                | Default     | Description                                                  |
| ------------------------ | ----------- | ------------------------------------------------------------ |
| `num_epochs`             | `400`       | Total number of training epochs                              |
| `batch_size`             | `16`        | Batch size                                                   |
| `lr`                     | `5e-4`      | Base learning rate                                           |
| `backbone_lr_scale`      | `0.1`       | Learning-rate scaling ratio for the backbone                 |
| `warmup_epochs`          | `20`        | Number of learning-rate warm-up epochs                       |
| `lambda_tversky`         | `1.5`       | Weight of the Tversky loss                                   |
| `tversky_alpha` / `beta` | `0.4 / 0.6` | Tversky false-positive / false-negative penalty coefficients |
| `amp`                    | `True`      | Whether to enable mixed-precision training                   |

### `EvalConfig`

| Parameter    | Default                        | Description                              |
| ------------ | ------------------------------ | ---------------------------------------- |
| `checkpoint` | `checkpoints/best_model02.pth` | Checkpoint path used for evaluation      |
| `output_dir` | `predictions`                  | Directory for saving prediction results  |
| `tta`        | `False`                        | Whether to enable test-time augmentation |

---

