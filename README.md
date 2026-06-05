# CAFASeg-Net

**CAFASeg-Net** 是一个面向内窥镜视频的神经组织分割框架，专为 **极端类别不平衡**（前景像素占比 < 5%）和 **小目标结构** 场景设计。

---

## 目录

- [框架概览](#框架概览)
- [模型架构](#模型架构)
- [目录结构](#目录结构)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [训练](#训练)
- [评估](#评估)
- [推理](#推理)
- [半监督扩展路线](#半监督扩展路线)
- [配置参数说明](#配置参数说明)
- [常见问题](#常见问题)

---

## 框架概览

```
原始标注数据 (data/)
       ↓
  prepare_dataset.py      ← 数据预处理：FOV裁剪 / 生成Mask / 划分train-val-test
       ↓
  PELD/labeled_images/    ← 标准化数据集 (512×512 PNG)
       ↓
    train.py              ← 主监督训练 + 半监督 PCCP 联合训练
       ↓
  checkpoints/            ← 模型权重
       ↓
  evaluate.py             ← 测试集指标评估 + 可视化
  inference.py            ← 单图 / 图像目录 / 视频推理
  video_infer_V2.py       ← 视频推理（高级版，含置信度过滤）
```

---

## 模型架构

### 主干网络：DSBA-Bone（异构双流边界感知骨干）

```
输入 (3×512×512)
    ├── 语义流 (Semantic Stream)
    │     └── ResNet34 (ImageNet 预训练)
    │           ├── Stage1: [B, 64,  H/4,  W/4]
    │           ├── Stage2: [B, 128, H/8,  W/8]
    │           ├── Stage3: [B, 256, H/16, W/16]
    │           └── Stage4: [B, 512, H/32, W/32]
    │
    └── 边界流 (Boundary Stream)
          └── 轻量卷积 + CrossStreamInjection
                ├── f1: [B, 64,  H/4, W/4]  ← 注入语义流 Stage1
                └── f2: [B, 128, H/8, W/8]  ← 注入语义流 Stage2
         ↓
    瓶颈层：ASPPLite (空洞空间金字塔池化，4分支: r=1,6,12,GAP)
         ↓
    解码器：4× DecoderBlock (CAFM 跨尺度自适应特征融合 + 双线性上采样)
         ↓
    精修：BERD (Sobel 梯度调制边界增强)
         ↓
    输出头 (1×512×512 Sigmoid)
```

### 核心模块

| 模块 | 全称 | 功能 |
|------|------|------|
| **DSBA-Bone** | Dual-Stream Boundary-Aware Backbone | 双流异构骨干，语义流 + 边界流跨流注入 |
| **ASPPLite** | Atrous Spatial Pyramid Pooling Lite | 多尺度上下文聚合，膨胀率 1/6/12 + 全局平均池化 |
| **CAFM** | Cross-scale Adaptive Feature Merging | 通道注意力 + 空间注意力门控融合 encoder-decoder 特征 |
| **BERD** | Boundary-Enhanced Refinement Decoder | Sobel 梯度检测 + 可学习 γ 调制边界响应 |

> 兼容备用骨干：`backbone="segformer"` (需安装 `transformers`)，`backbone="arfe"` (Gumbel-Softmax 动态路由)

---

## 目录结构

```
CAFASeg-Net/
├── config.py                  # 全局超参数配置中心
├── prepare_dataset.py         # 数据集预处理脚本
├── dataset.py                 # Dataset / DataLoader / 增强流水线
├── models.py                  # 模型架构定义
├── losses.py                  # 损失函数（Tversky / CCDice / PCCP）
├── metrics.py                 # 评估指标（Dice / IoU / HD95）
├── train.py                   # 训练主程序
├── evaluate.py                # 测试集评估与可视化
├── inference.py               # 单图/目录/视频推理
├── video_infer.py             # 视频推理（稳定版，ROI裁剪）
├── video_infer_V2.py          # 视频推理（完整版，含流推理和置信度过滤）
├── route1_pseudo_label.py     # 半监督路线一：伪标签法
├── route3_optical_flow.py     # 半监督路线三：光流时序一致性
└── requirements.txt           # 依赖清单

# 运行后自动生成：
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

checkpoints/                   # 模型权重保存目录
logs/                          # TensorBoard 训练日志
predictions/                   # evaluate.py 输出预测图
```

---

## 环境配置

### 1. 创建虚拟环境

```bash
conda create -n cafaseg python=3.10 -y
conda activate cafaseg
```

### 2. 安装 PyTorch（以 CUDA 11.8 为例）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

> 请根据实际 CUDA 版本至 [pytorch.org](https://pytorch.org/get-started/locally/) 选择对应命令。

### 3. 安装其余依赖

```bash
pip install -r requirements.txt
```

### 4. 验证安装

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python models.py   # 运行内置测试，输出 logits shape 和参数量
```

---

## 数据准备

### 原始数据格式

```
data/
├── images/
│   ├── case001.png
│   └── ...
└── labels/
    ├── case001.json    # LabelMe 格式，shapes[].points 多边形顶点
    └── ...
```

### 执行预处理

```bash
# 基础用法（使用默认路径和参数）
python prepare_dataset.py

# 指定路径 + 多进程加速 + 生成可视化验证图
python prepare_dataset.py --src /root/autodl-tmp/data --dst /root/autodl-tmp/PELD --workers 8 --vis

# 自定义划分比例（例如 7:2:1）
python prepare_dataset.py --split 0.7 0.2 0.1

# 自动向 config.py 追加测试集路径字段
python prepare_dataset.py --patch_cfg
```

**预处理流程：**
1. 扫描 `data/labels/` 并过滤空标注
2. 从 JSON 多边形生成二值 Mask（`fillPoly`）
3. 自动检测内窥镜 FOV 圆形区域，裁剪为最小外接正方形
4. 缩放至 512×512（图像用 `INTER_AREA`，Mask 用 `INTER_NEAREST`）
5. 按 `8:1:1` 随机划分并保存至 `PELD/labeled_images/`
6. 打印各划分的前景分布统计报告

### 诊断数据集类别不平衡

```python
from dataset import diagnose_dataset
diagnose_dataset("PELD/labeled_images/train/images",
                 "PELD/labeled_images/train/masks")
```

---

## 训练

### 基础训练（纯监督）

```bash
python train.py
```

### 常用参数

```bash
# 指定数据集根目录
python train.py --data_root /path/to/PELD

# 从断点恢复训练
python train.py --resume checkpoints/last_checkpoint.pth

# 修改批大小和学习率
python train.py --batch_size 8 --lr 3e-4

# 禁用 AMP 混合精度（调试用）
python train.py --no_amp
```

### 半监督联合训练（监督 + PCCP 无监督）

在 `config.py` 中配置无标签视频目录：

```python
unlabeled_dirs: List[str] = field(default_factory=lambda: [
    "unlabeled_videos/patient_A",
    "unlabeled_videos/patient_B",
])
```

训练程序会自动检测无标签数据集并启用 PCCP（物理驱动交叉对比伪监督）损失。

### 训练日志

- TensorBoard：`tensorboard --logdir logs/`
- Excel 训练记录：`logs/training_log.xlsx`（需安装 `openpyxl`）
- 控制台实时输出 Dice / Loss 表格

### 差分学习率策略（DSBA-Bone）

| 参数组 | 学习率 | 说明 |
|--------|--------|------|
| 预训练语义流（ResNet34）| `lr × backbone_lr_scale`（默认 5e-5）| 慢速微调，保护 ImageNet 先验 |
| 边界流 + 解码器 | `lr`（默认 5e-4）| 从零初始化，完整学习率收敛 |

---

## 评估

```bash
# 在验证集上评估（使用 config.py 中配置的 checkpoint）
python evaluate.py

# 指定权重和输出目录
python evaluate.py --checkpoint checkpoints/best_model.pth --vis_dir eval_vis

# 保存所有帧的可视化图
python evaluate.py --checkpoint best_model.pth --vis_all

# 仅计算指标，不生成可视化图
python evaluate.py --checkpoint best_model.pth --no_vis
```

### 输出指标

| 指标 | 说明 |
|------|------|
| mDSC | 平均 Dice 相似系数（前景专属） |
| mIoU | 平均交并比（前景专属） |
| HD95 | 平均 95% Hausdorff 距离（mm） |
| Precision | 精确率 = TP / (TP + FP) |
| Recall | 召回率 = TP / (TP + FN) |
| Pixel_Acc | 像素准确率 |
| FPS | 推理吞吐（帧/秒） |

### 可视化输出目录结构

```
eval_vis/
├── comparisons/       # 五联图：原图 | GT叠加 | 预测叠加 | 置信度热图 | 误差图
├── heatmaps/          # 独立置信度热图
├── summary/           # 最差/最佳样本汇总网格 + Dice 分布直方图
└── direct_overlay/    # GT轮廓(绿) vs 预测轮廓(洋红) 对比图
```

---

## 推理

### 单张图像

```bash
python inference.py --mode image \
    --input path/to/image.jpg \
    --checkpoint checkpoints/best_model.pth \
    --output_dir predictions/
```

### 图像目录批量推理

```bash
python inference.py --mode dir \
    --input path/to/images/ \
    --checkpoint checkpoints/best_model.pth \
    --output_dir predictions/
```

### 视频推理

```bash
# 稳定版（ROI裁剪加速 + 保序输出）
python video_infer.py --mode video \
    --input surgery_video.mp4 \
    --checkpoint checkpoints/best_model.pth \
    --output result.mp4

# 完整版（含置信度过滤 + 实时流推理）
python video_infer_V2.py --mode video \
    --input surgery_video.mp4 \
    --checkpoint checkpoints/best_model.pth \
    --output result.mp4

# 实时摄像头流（video_infer_V2.py 专属）
python video_infer_V2.py --mode stream \
    --input 0 \
    --checkpoint checkpoints/best_model.pth
```

---

## 半监督扩展路线

### 路线一：伪标签法 (`route1_pseudo_label.py`)

使用教师模型对无标签数据生成高置信度伪标签，再与人工标注混合训练。

```bash
# 步骤 1：生成伪标签（只需运行一次）
python route1_pseudo_label.py --step generate \
    --checkpoint checkpoints/best_model.pth \
    --output_dir pseudo_labels \
    --conf_high 0.90 --conf_low 0.10 \
    --min_fg 0.01 --max_fg 0.30

# 步骤 2：混合训练
python route1_pseudo_label.py --step train \
    --pseudo_dir pseudo_labels \
    --resume checkpoints/best_model.pth
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--conf_high` | 0.90 | 高置信前景阈值 |
| `--conf_low` | 0.10 | 高置信背景阈值 |
| `--min_fg` | 0.01 | 保留帧最小前景占比 |
| `--max_fg` | 0.30 | 保留帧最大前景占比（过滤异常帧）|
| `--pseudo_weight` | 0.50 | 伪标签损失权重（<1.0 反映不确定性）|

### 路线三：光流时序一致性 (`route3_optical_flow.py`)

利用 RAFT 光流对齐相邻帧预测，构建时序一致性约束。

```bash
# 步骤 1：离线预提取光流（推荐，节省训练时间）
python route3_optical_flow.py --step precompute_flow \
    --flow_dir flow_cache

# 步骤 2：启动训练
python route3_optical_flow.py --step train \
    --resume checkpoints/best_model.pth \
    --flow_dir flow_cache

# 可选：在线光流（无需预提取，但每步约多 50ms）
python route3_optical_flow.py --step train \
    --resume checkpoints/best_model.pth \
    --online_flow
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lambda_flow` | 0.01 | 光流时序损失权重 |
| `--flow_warmup` | 50 | 激活光流损失的起始 epoch |
| `--warp_conf` | 0.80 | 光流置信度阈值（低于此值的遮挡区域不计入损失）|

---

## 配置参数说明

所有超参数集中在 `config.py`，分为四个数据类：

### `DataConfig`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dataset_root` | `/root/autodl-tmp/PELD` | 数据集根目录 |
| `image_size` | `(512, 512)` | 模型输入尺寸 |
| `mask_threshold` | `0.35` | Mask 二值化阈值 |
| `num_classes` | `1` | 分割类别数（二分类取1）|

### `ModelConfig`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `backbone` | `"dsbabone"` | 骨干网络选择：`dsbabone` / `segformer` / `arfe` |
| `encoder_channels` | `(64,128,256,512)` | 编码器各阶段输出通道数 |
| `decoder_channels` | `(128,64,32,16)` | 解码器各阶段输出通道数 |

### `TrainConfig`（关键参数）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_epochs` | `400` | 总训练轮数 |
| `batch_size` | `16` | 批大小 |
| `lr` | `5e-4` | 基础学习率 |
| `backbone_lr_scale` | `0.1` | 骨干网络学习率缩放比 |
| `warmup_epochs` | `20` | 学习率预热轮数 |
| `lambda_tversky` | `1.5` | Tversky 损失权重 |
| `tversky_alpha` / `beta` | `0.4 / 0.6` | Tversky 假阳性/假阴性惩罚系数 |
| `amp` | `True` | 是否启用混合精度训练 |

### `EvalConfig`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `checkpoint` | `checkpoints/best_model02.pth` | 评估使用的权重路径 |
| `output_dir` | `predictions` | 预测结果保存目录 |
| `tta` | `False` | 是否启用测试时增强 |

---

## 常见问题

**Q: 训练时出现 `UserWarning: pretrained is deprecated`**
> 已在代码中更新为 `weights=ResNet34_Weights.DEFAULT` 新 API，此警告不影响运行。

**Q: CUDA Out of Memory**
> 减小 `config.py` 中的 `batch_size`（推荐最小 4），或在 `train.py` 中添加 `--batch_size 4`。

**Q: FOV 裁剪效果不佳（非圆形内窥镜图像）**
> `prepare_dataset.py` 中 FOV 检测有面积比保护（< 10% 全图时跳过裁剪），非圆形图像会自动跳过，保持原始比例。

**Q: 如何切换 SegFormer 骨干**
> 1. 安装 `pip install transformers`
> 2. 修改 `config.py`：`backbone = "segformer"`，`segformer_variant = "nvidia/mit-b2"`

**Q: `route3_optical_flow.py` 报 RAFT 未找到**
> 安装 `torchvision>=0.15.0` 后使用内置 `torchvision.models.optical_flow.raft_small`，或从 [RAFT 官方仓库](https://github.com/princeton-vl/RAFT) 安装。

**Q: HuggingFace 模型下载失败（国内环境）**
> `models.py` 和 `train.py` 已自动设置镜像源 `HF_ENDPOINT=https://hf-mirror.com`，无需手动配置。
