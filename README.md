# CAFASeg-Net

CAFASeg-Net is a PyTorch reference implementation of a boundary-aware binary segmentation network for foreground tissue segmentation. The repository contains the model definition, data processing utilities, loss functions, metric implementations, and scripts used to organize the experimental pipeline.

## Overview

The network combines a ResNet-style encoder with lightweight boundary modeling, multi-scale context aggregation, adaptive decoder fusion, and a boundary-enhanced refinement head. It is designed for binary medical image segmentation tasks where the foreground region is small, thin, or visually ambiguous.

## Main Components

- Boundary-aware encoder for extracting semantic and edge-sensitive features.
- ASPP-style bottleneck for multi-scale contextual representation.
- Cross-scale decoder fusion for recovering spatial details.
- Boundary-enhanced refinement before the prediction head.
- Dice, Tversky, BCE, connected-component-aware, and auxiliary supervision losses.
- Evaluation utilities for Dice, IoU, HD95, precision, recall, pixel accuracy, FPS, and component-level recall.
- Visualization helpers for overlays, heatmaps, error maps, and comparison panels.

## Repository Structure

```text
.
├── config.py        # Project configuration
├── dataset.py       # Dataset construction and augmentation utilities
├── evaluate.py      # Evaluation and visualization pipeline
├── inference.py     # Image, folder, and video inference utilities
├── losses.py        # Loss functions
├── metrics.py       # Segmentation metrics
├── models.py        # CAFASeg-Net architecture
├── train.py         # Training pipeline
├── requirements.txt # Python dependencies
└── README.md
```

## Environment

The implementation is based on Python and PyTorch. A typical environment includes:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU experiments, install the PyTorch build that matches the local CUDA runtime.

## Dataset Format

The data loader follows a binary segmentation layout:

```text
dataset_root/
├── train/
│   ├── images/
│   └── masks/
└── val/
    ├── images/
    └── masks/
```

Masks are treated as binary maps, where background is `0` and foreground is non-zero. Common image formats are supported, including JPG, PNG, BMP, TIFF, and TIF.

## Configuration

Main experiment settings are defined in `config.py`, including:

- Dataset paths and image size.
- Encoder and decoder channel widths.
- Training epochs, batch size, learning rate, warmup, and weight decay.
- Loss weights and foreground-balancing parameters.
- Checkpoint, logging, and prediction output directories.

## Model Architecture

The architecture is implemented in `models.py`. The main network class is `CAFASegNet`, which returns the primary prediction logits and, during training, auxiliary decoder outputs for deep supervision.

## Evaluation Metrics

The evaluation pipeline reports:

- Dice coefficient
- Intersection over Union
- HD95
- Precision
- Recall
- Pixel accuracy
- Component recall
- Parameter count
- FPS

## Checkpoints and Data

Datasets, model checkpoints, logs, prediction outputs, and generated visualizations are not included in this repository. The `.gitignore` file excludes common runtime artifacts and large generated files.

## Citation

If this repository is useful for your research, please cite the corresponding paper or project page.
