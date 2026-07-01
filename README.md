<img width="2943" height="1782" alt="image" src="https://github.com/user-attachments/assets/0c2d8efc-675b-4738-a46e-36c554cbc328" />

# CAFASeg-Net

CAFASeg-Net is a PyTorch implementation of a boundary-aware binary segmentation network designed for endoscopic-style medical image segmentation. The model combines semantic feature extraction, boundary-sensitive representation, multi-scale context aggregation, cross-scale attention fusion, and boundary-enhanced refinement to improve foreground localization in visually complex regions.

This repository is prepared as a clean research-code release for project presentation, paper submission, and GitHub documentation.

---

## Overview

Medical image segmentation often becomes challenging when target regions are small, low-contrast, fragmented, or surrounded by visually similar background tissue. Standard encoder-decoder networks can recover coarse object regions, but they may lose fine boundary details during repeated downsampling.

CAFASeg-Net addresses this issue by introducing a dual-stream encoder and a boundary-enhanced refinement strategy. The model learns both semantic context and boundary-aware cues, then fuses them through attention-guided decoder blocks to generate accurate binary segmentation masks.

The framework is designed for binary foreground-background segmentation tasks, where the model receives an RGB image and predicts a single-channel probability map.

---

## Main Features

- Boundary-aware encoder with semantic and edge-sensitive branches
- Lightweight multi-scale context aggregation module
- Cross-scale attention fusion between encoder and decoder features
- Boundary-enhanced refinement before final prediction
- Composite loss design for region overlap, hard examples, and structural consistency
- Evaluation utilities for Dice, IoU, HD95, component recall, and FPS analysis
- Visualization utilities for qualitative segmentation comparison
- Clean and modular PyTorch implementation

---

## Network Architecture

The overall workflow of CAFASeg-Net is shown below:

```text
Input Image
    |
    v
Dual-Stream Boundary-Aware Encoder
    |-- Semantic feature stream
    |-- Boundary feature stream
    |-- Gated multi-level feature fusion
    v
ASPP-Lite Context Module
    v
Cross-Scale Attention Decoder
    v
Boundary-Enhanced Refinement
    v
Prediction Head
    v
Binary Segmentation Map
```

The model produces a binary foreground probability map. During training, auxiliary decoder outputs are also generated to support deep supervision.

---

## Method Components

### Dual-Stream Boundary-Aware Encoder

The encoder contains a semantic stream and a boundary stream. The semantic stream extracts hierarchical image representations, while the boundary stream preserves edge-related spatial information. Features from both streams are fused at multiple scales through learnable gates.

### ASPP-Lite Context Module

A lightweight atrous spatial pyramid pooling module is used at the bottleneck stage. It aggregates local, dilated, and global contextual information while keeping the parameter cost moderate.

### Cross-Scale Attention Fusion

The decoder uses cross-scale attention fusion to combine high-level decoder features with corresponding encoder features. Channel attention, spatial attention, and gated fusion are used to improve semantic consistency and recover fine spatial details.

### Boundary-Enhanced Refinement

Before the final prediction head, a refinement module enhances boundary-sensitive regions. It incorporates gradient response, structure response, and local contrast information to strengthen uncertain object boundaries.

### Prediction Head

The final prediction head maps refined decoder features into a one-channel logit map. A sigmoid activation converts the logits into foreground probabilities.

---

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
├── assets/
│   └── results/
│       ├── result_01.png
│       ├── result_02.png
│       ├── result_03.png
│       ├── result_04.png
│       ├── result_05.png
│       └── result_06.png
└── README.md
```

---

## File Description

| File | Description |
| --- | --- |
| `config.py` | Configuration definitions for data paths, model settings, training parameters, evaluation options, and output folders. |
| `dataset.py` | Dataset loading utilities, image-mask pairing, preprocessing, augmentation, and data loader construction. |
| `models.py` | Main CAFASeg-Net architecture, including encoder, context module, decoder fusion, refinement module, and prediction head. |
| `losses.py` | Loss functions for segmentation overlap, foreground imbalance, hard examples, boundary quality, and auxiliary supervision. |
| `metrics.py` | Evaluation metrics including Dice, IoU, HD95, component recall, and metric aggregation utilities. |
| `train.py` | Training pipeline structure, optimizer setup, scheduler logic, validation loop, checkpoint handling, and logging utilities. |
| `evaluate.py` | Evaluation utilities for metric calculation, prediction analysis, FPS estimation, and qualitative result export. |
| `inference.py` | Inference utilities for image-level, folder-level, and video-level prediction workflows. |
| `requirements.txt` | Python dependencies used by the implementation. |
| `.gitignore` | Ignore rules for datasets, checkpoints, logs, generated outputs, caches, and local environments. |

---

## Results Visualization

<img width="2603" height="1092" alt="image" src="https://github.com/user-attachments/assets/006a5c75-ece7-41f5-8eb8-5c457312cb60" />


## Quantitative Evaluation

CAFASeg-Net can be evaluated using commonly used binary segmentation metrics:

| Metric | Description |
| --- | --- |
| Dice | Measures region overlap between prediction and ground truth. |
| IoU | Measures intersection-over-union segmentation quality. |
| HD95 | Measures the 95th percentile Hausdorff distance for boundary accuracy. |
| Component Recall | Measures the ability to recover separated foreground components. |
| FPS | Reports inference speed under the selected runtime environment. |
<img width="2901" height="690" alt="image" src="https://github.com/user-attachments/assets/c6235220-3104-4194-ac74-00d12f9cfcd9" />

These metrics provide complementary views of segmentation quality, including region-level accuracy, boundary-level quality, object-level completeness, and computational efficiency.

---

## Dataset Format

The implementation follows a paired image-mask dataset organization:

```text
dataset/
├── images/
│   ├── sample_001.png
│   ├── sample_002.png
│   └── ...
└── masks/
    ├── sample_001.png
    ├── sample_002.png
    └── ...
```

Each image should have a corresponding binary mask with the same file stem. The mask is treated as a foreground-background annotation.

---

## Dependencies

The implementation is based on Python and PyTorch. Main dependencies include:

- Python
- PyTorch
- NumPy
- OpenCV
- Albumentations
- scikit-image
- tqdm
- matplotlib

Install the required packages with:

```bash
pip install -r requirements.txt
```

---

## Implementation Notes

This repository focuses on presenting the model implementation and research structure. Dataset files, trained checkpoints, experiment logs, and private runtime assets are not included.

The codebase is organized for readability and modular inspection. Core architecture components can be found in `models.py`, while losses and metrics are separated into independent files for easier review.

---

## Citation

If this repository is useful for your research, please cite it as:

---

## License

This project is released for academic research and educational use. Please refer to the repository license file for detailed usage terms.
