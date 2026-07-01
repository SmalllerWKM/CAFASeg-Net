# CAFASeg-Net

CAFASeg-Net is a PyTorch reference implementation of a boundary-aware binary segmentation framework for foreground tissue segmentation in endoscopic-style medical imagery. The project is organized around a compact encoder-decoder architecture that combines semantic representation learning, boundary-sensitive feature extraction, multi-scale context aggregation, cross-scale attention fusion, and boundary-enhanced refinement.

The repository is intended to document the implementation structure used for research presentation, ablation discussion, and code inspection. Datasets, trained checkpoints, runtime experiment assets, and private experiment launch settings are not included in this package.

## Table of Contents

- [Project Overview](#project-overview)
- [Research Motivation](#research-motivation)
- [Method Summary](#method-summary)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Core Modules](#core-modules)
- [Configuration](#configuration)
- [Dataset Organization](#dataset-organization)
- [Data Processing and Augmentation](#data-processing-and-augmentation)
- [Training Design](#training-design)
- [Loss Functions](#loss-functions)
- [Evaluation Protocol](#evaluation-protocol)
- [Inference Utilities](#inference-utilities)
- [Visualization Utilities](#visualization-utilities)
- [Dependencies](#dependencies)
- [Repository Scope](#repository-scope)
- [Suggested GitHub Layout](#suggested-github-layout)
- [Citation](#citation)
- [License](#license)

## Project Overview

CAFASeg-Net focuses on binary segmentation, where each pixel is classified as either background or foreground. The implementation is designed for scenarios in which target regions may be small, low-contrast, fragmented, partially occluded, or surrounded by visually similar tissue structures.

The codebase contains:

- A ResNet-style semantic encoder enhanced with a parallel boundary stream.
- A lightweight multi-scale context bottleneck.
- Cross-scale feature fusion blocks for encoder-decoder interaction.
- A boundary-enhanced refinement module before the final prediction head.
- Loss functions for region overlap, foreground imbalance, hard examples, and connected-component consistency.
- Dataset utilities for paired image-mask loading and common augmentation strategies.
- Evaluation metrics for overlap quality, boundary distance, component coverage, and runtime profiling.
- Visualization helpers for qualitative segmentation analysis.

The implementation is written in a modular style so that architecture components, losses, metrics, and data utilities can be inspected or reused independently.

## Research Motivation

Binary medical image segmentation can be difficult when the foreground occupies a small fraction of the image or when its boundary is weak. Standard encoder-decoder networks often recover coarse region structure but may miss thin structures, small detached components, or ambiguous edges. CAFASeg-Net addresses these problems by combining semantic features with boundary-aware cues at multiple stages of the network.

The design follows three main ideas:

1. **Boundary-sensitive representation.** A dedicated boundary stream complements the semantic encoder, allowing the network to preserve edge-related structures that may be weakened by repeated downsampling.
2. **Context-aware decoding.** Multi-scale context aggregation provides a wider receptive field before the decoder reconstructs high-resolution predictions.
3. **Refinement near uncertain edges.** The final decoder output is refined using boundary, structure, and local contrast responses before producing the segmentation logits.

## Method Summary

Given an RGB input image, CAFASeg-Net generates a single-channel binary segmentation probability map. The main prediction flow is:

```text
Input Image
    |
    v
DSBA-Bone Encoder
    |-- semantic stream
    |-- boundary stream
    |-- gated fusion at four scales
    v
ASPP-Lite Bottleneck
    v
Cross-Scale Decoder Blocks
    v
Boundary-Enhanced Refinement
    v
Prediction Head
    v
Binary Probability Map
```

During training, the model also produces auxiliary decoder outputs for deep supervision. These auxiliary outputs are used by the training loss to encourage intermediate decoder stages to learn meaningful segmentation representations.

## Architecture

The main architecture is implemented in `models.py` under the `CAFASegNet` class.

### 1. DSBA-Bone Encoder

`DSBABone` is the encoder module. It contains two parallel streams:

- **Semantic stream:** a ResNet-34 backbone that extracts hierarchical semantic features.
- **Boundary stream:** a lightweight convolutional branch that preserves edge-aware features at corresponding scales.

At each encoder stage, semantic and boundary features are combined through a learned gate. The gate receives concatenated features and predicts a fusion weight, allowing the encoder to adaptively balance semantic and boundary information.

The encoder returns four feature maps:

```text
f1: low-level feature map with fine spatial detail
f2: intermediate feature map
f3: high-level feature map
f4: deepest semantic-boundary feature map
```

These features are passed to the bottleneck and decoder.

### 2. ASPP-Lite Bottleneck

`ASPPLite` performs lightweight atrous spatial pyramid pooling. It uses multiple branches with different receptive fields:

- Local convolution branch.
- Dilated convolution branch with medium dilation.
- Dilated convolution branch with larger dilation.
- Global pooling branch.

The branch outputs are concatenated, projected back to the original channel dimension, and added to the input through a residual connection. This module helps the network capture both local detail and larger contextual cues.

### 3. CAFM Decoder Fusion

`CAFM` is the cross-scale attention fusion module used inside decoder blocks. It aligns encoder and decoder features to the same channel dimension, then applies:

- Channel attention over encoder features.
- Spatial attention over decoder features.
- A learned fusion gate between the two feature sources.
- A final convolutional refinement layer.

This design helps the decoder recover spatial details while retaining high-level semantic consistency.

### 4. Decoder Blocks

Each `DecoderBlock` upsamples the decoder feature map and fuses it with the corresponding encoder feature map through `CAFM`. The decoder gradually reconstructs higher-resolution features from the bottleneck representation.

The default decoder channel progression is:

```text
512 -> 128 -> 64 -> 32 -> 16
```

### 5. BERD Refinement

`BERD` is the boundary-enhanced refinement module applied near the end of the network. It combines three structural cues:

- Sobel-like gradient response.
- Structure tensor anisotropy response.
- Local center-surround contrast response.

A learned weighting mechanism combines these cues, and a gated residual connection injects the refined boundary information into the final decoder feature map.

### 6. Prediction Head

The prediction head is a `1x1` convolution that maps the refined decoder feature map to a single-channel logit map. The sigmoid of this logit map is used as the binary foreground probability.

The model output dictionary contains:

```text
logits: raw segmentation logits
pred: sigmoid probability map
aux_logits3: auxiliary output from a deeper decoder stage, available during training
aux_logits2: auxiliary output from an intermediate decoder stage, available during training
```

## Repository Structure

```text
CAFASeg-Net/
├── config.py
├── dataset.py
├── evaluate.py
├── inference.py
├── losses.py
├── metrics.py
├── models.py
├── train.py
├── requirements.txt
├── .gitignore
└── README.md
```

### File Descriptions

| File | Description |
| --- | --- |
| `config.py` | Central configuration dataclasses for data paths, model channels, training hyperparameters, evaluation settings, and output directories. |
| `dataset.py` | Dataset loading, image-mask pairing, mask binarization, augmentation pipelines, foreground-aware cropping, copy-paste augmentation, and data loader construction. |
| `models.py` | CAFASeg-Net architecture, including DSBA-Bone, ASPP-Lite, CAFM decoder fusion, BERD refinement, and the final segmentation network. |
| `losses.py` | Segmentation losses including focal Tversky loss, boundary-weighted Dice loss, connected-component Dice loss, auxiliary supervision, and composite training objective. |
| `metrics.py` | Metric implementations for Dice, IoU, HD95, component recall, and aggregated metric reporting. |
| `train.py` | Training pipeline structure, optimizer and scheduler setup, mixed precision support, validation loop, checkpoint handling, EMA utilities, and logging helpers. |
| `evaluate.py` | Evaluation and qualitative analysis utilities, including metrics, FPS measurement, visualization export, and summary reporting. |
| `inference.py` | Image, directory, and video inference utilities with preprocessing, postprocessing, connected-component filtering, and overlay generation. |
| `requirements.txt` | Python package dependencies used by the implementation. |
| `.gitignore` | Excludes checkpoints, logs, generated predictions, datasets, videos, cache files, and environment folders. |

## Core Modules

### `config.py`

The configuration file defines four dataclass groups:

- `DataConfig`: dataset root, image and mask folder names, unlabeled folders, image size, mask threshold, and number of output classes.
- `ModelConfig`: encoder channels, decoder channels, input channels, boundary refinement initialization, and head width.
- `TrainConfig`: random seed, epoch count, batch size, optimizer settings, learning-rate schedule, loss weights, foreground-balancing parameters, checkpoint paths, and logging paths.
- `EvalConfig`: checkpoint path, prediction output directory, test-time augmentation flag, and HD95 spacing.

A top-level `Config` object combines these groups and creates output directories for checkpoints, logs, and predictions.

### `dataset.py`

The dataset module provides paired binary segmentation data loading. It includes utilities for:

- Collecting images by extension.
- Matching image files with mask files by stem and extension.
- Reading RGB images and grayscale masks with OpenCV.
- Converting grayscale masks into binary foreground-background maps.
- Building training and validation transforms with Albumentations.
- Applying foreground-aware cropping to reduce empty-background dominance.
- Applying copy-paste foreground augmentation to increase small-object diversity.
- Building PyTorch `DataLoader` objects for supervised training and validation.
- Diagnosing dataset folder contents and missing masks.

Supported image extensions include:

```text
.jpg, .jpeg, .png, .bmp, .tiff, .tif
```

### `models.py`

The model module contains the neural network implementation. Important classes include:

| Class | Role |
| --- | --- |
| `ConvBNAct` | Basic convolution, batch normalization, and activation block. |
| `DepthwiseSeparableConv` | Depthwise separable convolution utility. |
| `DSBABone` | Dual-stream semantic-boundary encoder. |
| `ASPPLite` | Lightweight multi-scale context bottleneck. |
| `CAFM` | Cross-scale attention fusion module for decoder skip connections. |
| `BERD` | Boundary-enhanced refinement module. |
| `DecoderBlock` | Upsampling and encoder-decoder fusion block. |
| `CAFASegNet` | Full segmentation network. |

### `losses.py`

The loss module includes several objectives for binary segmentation:

| Loss | Purpose |
| --- | --- |
| `FocalTverskyLoss` | Handles foreground-background imbalance by emphasizing false positives and false negatives through Tversky weighting. |
| `BoundaryWeightedDiceLoss` | Applies region overlap supervision with additional emphasis around foreground boundaries and hard pixels. |
| `PCCPLoss` | Provides a contrastive and pseudo-label consistency formulation for feature-level supervision. |
| `CCDiceLoss` | Encourages connected-component-aware segmentation behavior. |
| `CAFASegNetLoss` | Combines main segmentation losses and auxiliary decoder supervision into a composite training objective. |

### `metrics.py`

The metric module evaluates binary segmentation predictions with:

- Dice coefficient.
- Intersection over Union.
- HD95 boundary distance.
- Component recall.
- Aggregated mean and standard deviation reporting.
- Handling of true-negative frames and false-positive-only frames.

### `train.py`

The training module contains the experiment loop design, including:

- Random seed setup.
- Model, optimizer, scheduler, and gradient scaler construction.
- Mixed precision training support.
- Cosine learning-rate scheduling with warmup.
- Exponential moving average model wrapper.
- Training and validation loops.
- Checkpoint saving logic.
- Console, TensorBoard, and spreadsheet-style metric logging.

### `evaluate.py`

The evaluation module contains:

- Model checkpoint loading logic.
- Validation-set prediction and metric aggregation.
- HD95 calculation support.
- Parameter count reporting.
- FPS measurement utilities.
- Qualitative visualization export.
- Summary table printing.

### `inference.py`

The inference module contains utilities for:

- Loading images.
- Applying normalization and resizing transforms.
- Running model prediction.
- Thresholding probability maps.
- Removing small connected components.
- Keeping the largest connected component.
- Producing image overlays.
- Processing single images, image folders, and videos.

## Configuration

The default configuration values are stored directly in `config.py`. The main groups are summarized below.

### Data Settings

| Setting | Meaning |
| --- | --- |
| `dataset_root` | Root directory of the segmentation dataset. |
| `train_img_dir` | Relative path to training images. |
| `train_msk_dir` | Relative path to training masks. |
| `val_img_dir` | Relative path to validation images. |
| `val_msk_dir` | Relative path to validation masks. |
| `image_size` | Target input resolution used by the pipeline. |
| `mask_threshold` | Threshold for converting grayscale masks into binary masks. |
| `num_classes` | Number of output channels. The default is one for binary segmentation. |

### Model Settings

| Setting | Meaning |
| --- | --- |
| `encoder_channels` | Channel dimensions produced by the encoder stages. |
| `decoder_channels` | Channel dimensions used by the decoder stages. |
| `in_channels` | Number of input image channels. |
| `berd_gamma_init` | Initial scale used by the boundary refinement branch. |
| `head_channels` | Channel width associated with the final prediction head design. |

### Training Settings

| Setting | Meaning |
| --- | --- |
| `seed` | Random seed for reproducibility control. |
| `num_epochs` | Number of training epochs in the experiment configuration. |
| `batch_size` | Batch size used by data loaders. |
| `num_workers` | Worker count for data loading. |
| `device` | Preferred compute device. |
| `amp` | Mixed precision training flag. |
| `lr` | Initial learning rate. |
| `weight_decay` | Optimizer weight decay. |
| `grad_clip` | Gradient clipping value. |
| `backbone_lr_scale` | Learning-rate scale applied to backbone parameters. |
| `warmup_epochs` | Number of warmup epochs before cosine annealing. |
| `lr_min` | Minimum learning rate in the cosine schedule. |
| `lambda_bce` | Weight for BCE-style foreground supervision. |
| `lambda_tversky` | Weight for Tversky-style overlap supervision. |
| `lambda_deep_sup3` | Auxiliary supervision weight for the deeper decoder output. |
| `lambda_deep_sup2` | Auxiliary supervision weight for the intermediate decoder output. |
| `lambda_cc_dice` | Connected-component-aware loss weight. |

### Evaluation Settings

| Setting | Meaning |
| --- | --- |
| `checkpoint` | Default checkpoint path expected by evaluation and inference utilities. |
| `output_dir` | Default directory for generated predictions. |
| `tta` | Test-time augmentation flag. |
| `hd95_spacing` | Pixel spacing used for HD95 calculation. |

## Dataset Organization

The data loader expects a paired image-mask layout for supervised binary segmentation:

```text
dataset_root/
├── train/
│   ├── images/
│   │   ├── sample_001.png
│   │   ├── sample_002.png
│   │   └── ...
│   └── masks/
│       ├── sample_001.png
│       ├── sample_002.png
│       └── ...
└── val/
    ├── images/
    │   ├── sample_101.png
    │   ├── sample_102.png
    │   └── ...
    └── masks/
        ├── sample_101.png
        ├── sample_102.png
        └── ...
```

Image and mask files are paired by filename stem. Masks are loaded as grayscale images and converted into binary maps:

```text
background: 0
foreground: non-zero value after thresholding
```

For reliable pairing, each image should have a corresponding mask with the same stem. The extension may differ if the stem is unchanged.

## Data Processing and Augmentation

The training pipeline uses Albumentations for image and mask transforms. The main training augmentations include:

- Resizing.
- Horizontal and vertical flipping.
- Affine translation, scaling, and rotation.
- Color jittering.
- CLAHE enhancement.
- Gaussian blur.
- Gaussian noise.
- Elastic deformation.
- ImageNet-style normalization.
- Tensor conversion.

The validation transform applies resizing, normalization, and tensor conversion without random perturbation.

The dataset utilities also include foreground-oriented operations:

- **Foreground-aware crop:** increases the probability that a crop contains foreground pixels.
- **Copy-paste foreground augmentation:** copies connected foreground components from one sample into another sample.
- **Mask thresholding:** converts soft or grayscale masks into binary labels.
- **Dataset diagnostics:** checks image-mask pairing and dataset consistency.

## Training Design

The training code is structured around supervised binary segmentation. The design includes:

- Construction of labeled training and validation loaders.
- Model initialization from the architecture configuration.
- AdamW-style optimization settings.
- Differential learning-rate scaling for backbone parameters.
- Warmup followed by cosine annealing.
- Mixed precision support with automatic gradient scaling.
- Gradient clipping for stable optimization.
- Deep supervision through auxiliary decoder outputs.
- Exponential moving average of model parameters.
- Periodic validation and best-checkpoint tracking.
- TensorBoard-compatible scalar logging.
- Spreadsheet-style validation logging when `openpyxl` is available.

The training script is kept as an implementation reference. Runtime datasets, trained weights, and experiment-specific execution assets are outside the repository package.

## Loss Functions

The composite loss used by the training pipeline combines several complementary terms.

### Focal Tversky Loss

Focal Tversky loss is used to handle foreground-background imbalance. The Tversky formulation separately weights false positives and false negatives, while the focal exponent emphasizes difficult samples.

### Boundary-Weighted Dice Loss

Boundary-weighted Dice loss increases supervision strength around uncertain or structurally important pixels. This is useful when foreground regions have thin, irregular, or ambiguous boundaries.

### Connected-Component Dice Loss

Connected-component-aware Dice loss focuses on object-level completeness. It encourages the model to recover separated foreground components rather than only optimizing global overlap.

### Auxiliary Supervision

The decoder produces auxiliary logits during training. These outputs provide additional supervision at intermediate decoder stages and help stabilize feature learning.

### Composite Objective

`CAFASegNetLoss` combines the main segmentation loss terms and auxiliary losses according to the weights defined in `config.py`.

## Evaluation Protocol

The evaluation utilities support both quantitative and qualitative analysis.

### Quantitative Metrics

| Metric | Interpretation |
| --- | --- |
| Dice | Measures overlap between prediction and ground truth. Higher is better. |
| IoU | Measures intersection-over-union overlap. Higher is better. |
| HD95 | Measures robust boundary distance using the 95th percentile Hausdorff distance. Lower is better. |
| Precision | Measures the proportion of predicted foreground pixels that are correct. Higher is better. |
| Recall | Measures the proportion of ground-truth foreground pixels recovered by the model. Higher is better. |
| Pixel accuracy | Measures the proportion of correctly classified pixels. Higher is better. |
| Component recall | Measures how many ground-truth connected components are detected. Higher is better. |
| FPS | Estimates inference speed. Higher is better. |
| Parameter count | Reports model size in millions of parameters. |

### Empty-Frame Handling

Medical video or frame-based datasets may contain images with no foreground target. The metric implementation accounts for these cases:

- Empty prediction with empty target is treated as a true-negative frame.
- Prediction on an empty-target frame may be skipped for overlap aggregation.
- Infinite HD95 cases are counted separately.

This avoids allowing empty frames to dominate the average overlap score.

## Inference Utilities

The inference module provides reusable utilities for applying a trained model to different input formats:

- Single image inference.
- Directory-level image inference.
- Video-frame inference.
- Probability thresholding.
- Small connected-component filtering.
- Largest-component postprocessing.
- Overlay visualization.
- Optional comparison with a ground-truth mask for single images or folders.

The postprocessing stage can remove small isolated predictions and preserve the most prominent connected foreground region.

## Visualization Utilities

The evaluation and inference scripts include visualization helpers for qualitative analysis:

- Direct mask overlays on the input image.
- Prediction heatmaps.
- Error maps showing false positives and false negatives.
- Multi-panel comparison figures.
- Summary grids for best and worst cases.
- Dice distribution histograms.

Generated visualizations are treated as runtime artifacts and are excluded from the repository by `.gitignore`.

## Dependencies

The implementation relies on the following major Python packages:

| Package | Purpose |
| --- | --- |
| `torch` | Neural network implementation and training. |
| `torchvision` | ResNet backbone and image model utilities. |
| `numpy` | Numerical operations. |
| `opencv-python` | Image and video IO, connected components, and visualization utilities. |
| `albumentations` | Data augmentation and preprocessing. |
| `scipy` | HD95 distance calculation. |
| `matplotlib` | Visualization export. |
| `tqdm` | Progress display. |
| `tensorboard` | Training scalar logging. |
| `tabulate` | Console metric tables. |
| `openpyxl` | Spreadsheet-style metric logging. |

See `requirements.txt` for package constraints.

## Repository Scope

This repository contains the research code structure and implementation modules. The following items are not included:

- Original datasets.
- Processed datasets.
- Trained checkpoints.
- Experiment logs.
- Prediction outputs.
- Visualization outputs.
- Local machine paths.
- Private execution settings.

The `.gitignore` file excludes common runtime artifacts such as:

```text
checkpoints/
logs/
predictions/
eval_vis/
outputs/
runs/
data/
datasets/
*.pth
*.pt
*.ckpt
*.onnx
*.engine
```

## Suggested GitHub Layout

For a clean public repository, the following layout is recommended:

```text
CAFASeg-Net/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py
├── dataset.py
├── models.py
├── losses.py
├── metrics.py
├── train.py
├── evaluate.py
└── inference.py
```

Large assets should be stored outside GitHub or managed through a dedicated artifact storage system. Generated outputs should not be committed.

## Citation

If this repository is useful for your research, please cite the corresponding paper or project page.



## License

Please add a license file before public distribution if the repository is intended for reuse. Common choices include MIT, Apache-2.0, and BSD-3-Clause. For research-only distribution, provide the appropriate usage terms in a separate `LICENSE` file.
