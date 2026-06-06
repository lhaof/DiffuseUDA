# Experimental Details

This README summarizes the dataset splits, training stages, and task-specific hyperparameters used for the unsupervised domain adaptation (UDA) experiments in medical image segmentation.

## 1. Overview

The experiments evaluate a diffusion-based generative UDA framework across three representative medical image segmentation settings:

- **MMWHS**: cross-modality cardiac tissue segmentation between CT and MR.
- **BraTS20**: cross-modality brain tumor segmentation between FLAIR and T2.
- **Fetal Brain**: cross-center fetal brain tissue segmentation between Atlas and FeTA datasets.

These tasks cover different anatomical regions, imaging modalities, and domain-shift patterns. Each experiment follows a staged training pipeline consisting of an initial segmentation model, a conditional diffusion model, and a second-stage segmentation model trained with generated data.

## 2. Dataset Splits

| Dataset | Modality / Task | Source | Target | Validation | Test | Iterative source data |
|---|---:|---:|---:|---:|---:|---|
| MMWHS | CT | 16 | 0 | 0 | 0 | 16 source + 13 generated |
| MMWHS | CT → MR, MR | 0 | 13 | 3 | 4 | 0 |
| MMWHS | MR | 13 | 0 | 0 | 0 | 13 source + 16 generated |
| MMWHS | MR → CT, CT | 0 | 16 | 3 | 4 | 0 |
| BraTS20 | FLAIR | 143 | 0 | 0 | 0 | 143 source + 143 generated |
| BraTS20 | FLAIR → T2, T2 | 0 | 143 | 41 | 42 | 0 |
| BraTS20 | T2 | 143 | 0 | 0 | 0 | 143 source + 143 generated |
| BraTS20 | T2 → FLAIR, FLAIR | 0 | 143 | 41 | 42 | 0 |
| Fetal Brain | T2 Atlas | 47 | 0 | 0 | 0 | 47 source + 40 target |
| Fetal Brain | T2 FeTA | 0 | 40 | 40 | 40 | 0 |

## 3. General Training Pipeline

Each UDA experiment is organized into three stages.

### Stage 1: Initial segmentation model

A segmentation model is trained using the available source data and target-domain data according to the UDA setting. The segmentation model used in the experiments is **ASC**.

### Stage 2: Conditional diffusion model

A conditional diffusion model is trained to generate image-mask pairs. The model is a **U-Net with one residual block**. Sampling conditions include real labels from the source domain and pseudo-labels from the target domain.

### Stage 3: Segmentation model with generated data

The segmentation model is further trained using source data augmented with diffusion-generated samples. Target-domain data are used according to the corresponding UDA training protocol.

## 4. Task-specific Settings

### 4.1 FeTA benchmark

| Parameter | Value |
|---|---|
| Pre-processing | Resize images and masks to 128 × 128 × 128 voxels |
| Source-domain data | 47 |
| Target-domain data | 40 |

#### Stage 1: Segmentation model

| Parameter | Value |
|---|---|
| Model | ASC |
| Optimizer | Adam |
| Learning rate | 2 × 10⁻³ |
| Batch size | 4, with 2 source and 2 target samples |
| Epochs | 100 |
| EMA decay | 0.99 |
| Model channels | 32 |
| Input size | 128 × 128 × 128 |

#### Stage 2: Diffusion model

| Parameter | Value |
|---|---|
| Model | U-Net with one residual block |
| Training data | 47 + 40 image-mask pairs |
| Optimizer | Adam |
| Learning rate | 1 × 10⁻⁵ |
| Batch size | 1 |
| Iterations | 60,000 |
| EMA decay | 0.995 |
| Model channels | 64 |
| Input size | 128 × 128 × 128 |
| Timesteps | 250 |
| Sampling conditions | Real labels from source (47), pseudo-labels from target (40) |
| Sample saving frequency | Every 1,000 iterations |
| Gradient accumulation | Every 2 steps |

#### Stage 3: Segmentation model

| Parameter | Value |
|---|---|
| Model | ASC |
| Source-domain data | 47 + 87 × 4 generated samples |
| Target-domain data | 40 |
| Optimizer | Adam |
| Learning rate | 1 × 10⁻⁴ |
| Batch size | 4, with 2 source and 2 target samples |
| Epochs | 15 |
| EMA decay | 0.99 |
| Model channels | 32 |
| Input size | 128 × 128 × 128 |

**Execution order:** Stage 1 → Stage 2 → Stage 3.

---

### 4.2 MMWHS MR-to-CT benchmark

| Parameter | Value |
|---|---|
| Pre-processing | Resize images and masks to 144 × 144 × 144 voxels |
| Source-domain data | 16 |
| Target-domain data | 13 |

#### Stage 1: Segmentation model

| Parameter | Value |
|---|---|
| Model | ASC |
| Optimizer | Adam |
| Learning rate | 5 × 10⁻³ |
| Batch size | 4, with 2 source and 2 target samples |
| Epochs | 100 |
| EMA decay | 0.99 |
| Model channels | 32 |
| Input size | 128 × 128 × 128, cropped like nnUNet |

#### Stage 2: Diffusion model

| Parameter | Value |
|---|---|
| Model | U-Net with one residual block |
| Training data | 16 + 13 image-mask pairs |
| Optimizer | Adam |
| Learning rate | 1 × 10⁻⁵ |
| Batch size | 1 |
| Iterations | 60,000 |
| EMA decay | 0.995 |
| Model channels | 64 |
| Input size | 144 × 144 × 144 |
| Timesteps | 250 |
| Sampling conditions | Real labels from source (16), pseudo-labels from target (13) |
| Sample saving frequency | Every 1,000 iterations |
| Gradient accumulation | Every 2 steps |

#### Stage 3: Segmentation model

| Parameter | Value |
|---|---|
| Model | ASC |
| Source-domain data | 16 + 29 × 4 generated samples |
| Target-domain data | 13 |
| Optimizer | Adam |
| Learning rate | 2 × 10⁻³ |
| Batch size | 4, with 2 source and 2 target samples |
| Epochs | 30 |
| EMA decay | 0.99 |
| Model channels | 32 |
| Input size | 128 × 128 × 128, cropped like nnUNet |

**Execution order:** Stage 1 → Stage 2 → Stage 3.

---

### 4.3 MMWHS CT-to-MR benchmark

| Parameter | Value |
|---|---|
| Pre-processing | Resize images and masks to 144 × 144 × 144 voxels |
| Source-domain data | 16 |
| Target-domain data | 13 |

#### Stage 1: Segmentation model

| Parameter | Value |
|---|---|
| Model | ASC |
| Optimizer | Adam |
| Learning rate | 5 × 10⁻³ |
| Batch size | 4, with 2 source and 2 target samples |
| Epochs | 100 |
| EMA decay | 0.99 |
| Model channels | 32 |
| Input size | 128 × 128 × 128, cropped like nnUNet |

#### Stage 2: Diffusion model

| Parameter | Value |
|---|---|
| Model | U-Net with one residual block |
| Training data | 16 + 13 image-mask pairs |
| Optimizer | Adam |
| Learning rate | 1 × 10⁻⁵ |
| Batch size | 1 |
| Iterations | 60,000 |
| EMA decay | 0.995 |
| Model channels | 64 |
| Input size | 144 × 144 × 144 |
| Timesteps | 250 |
| Sampling conditions | Real labels from source (16), pseudo-labels from target (13) |
| Sample saving frequency | Every 1,000 iterations |
| Gradient accumulation | Every 2 steps |

#### Stage 3: Segmentation model

| Parameter | Value |
|---|---|
| Model | ASC |
| Source-domain data | 16 + 29 × 4 generated samples |
| Target-domain data | 13 |
| Optimizer | Adam |
| Learning rate | 2 × 10⁻³ |
| Batch size | 4, with 2 source and 2 target samples |
| Epochs | 30 |
| EMA decay | 0.99 |
| Model channels | 32 |
| Input size | 128 × 128 × 128, cropped like nnUNet |

**Execution order:** Stage 1 → Stage 2 → Stage 3 → Stage 2 → Stage 3.

---

### 4.4 BraTS20 FLAIR-to-T2 benchmark

| Parameter | Value |
|---|---|
| Pre-processing | Resize images and masks to 64 × 128 × 128 voxels |
| Source-domain data | 143 |
| Target-domain data | 143 |

#### Stage 1: Segmentation model

| Parameter | Value |
|---|---|
| Model | ASC |
| Optimizer | Adam |
| Learning rate | 8 × 10⁻⁴ |
| Batch size | 4, with 2 source and 2 target samples |
| Epochs | 150 |
| EMA decay | 0.99 |
| Model channels | 32 |
| Input size | 64 × 128 × 128 |

#### Stage 2: Diffusion model

| Parameter | Value |
|---|---|
| Model | U-Net with one residual block |
| Training data | 143 + 143 image-mask pairs |
| Optimizer | Adam |
| Learning rate | 1 × 10⁻⁵ |
| Batch size | 1 |
| Iterations | 60,000 |
| EMA decay | 0.995 |
| Model channels | 128 |
| Input size | 64 × 128 × 128 |
| Timesteps | 250 |
| Sampling conditions | Real labels from source (143), pseudo-labels from target (143) |
| Sample saving frequency | Every 1,000 iterations |
| Gradient accumulation | Every 2 steps |

#### Stage 3: Segmentation model

| Parameter | Value |
|---|---|
| Model | ASC |
| Source-domain data | 143 + 143 generated samples |
| Target-domain data | 143 |
| Optimizer | Adam |
| Learning rate | 8 × 10⁻⁴ |
| Batch size | 4, with 2 source and 2 target samples |
| Epochs | 150 |
| EMA decay | 0.99 |
| Model channels | 32 |
| Input size | 64 × 128 × 128 |

**Execution order:** Stage 1 → Stage 2 → Stage 3 → Stage 2 → Stage 3.

---

### 4.5 BraTS20 T2-to-FLAIR benchmark

| Parameter | Value |
|---|---|
| Pre-processing | Resize images and masks to 64 × 128 × 128 voxels |
| Source-domain data | 143 |
| Target-domain data | 143 |

#### Stage 1: Segmentation model

| Parameter | Value |
|---|---|
| Model | ASC |
| Optimizer | Adam |
| Learning rate | 8 × 10⁻⁴ |
| Batch size | 4, with 2 source and 2 target samples |
| Epochs | 150 |
| EMA decay | 0.99 |
| Model channels | 32 |
| Input size | 64 × 128 × 128 |

#### Stage 2: Diffusion model

| Parameter | Value |
|---|---|
| Model | U-Net with one residual block |
| Training data | 143 + 143 image-mask pairs |
| Optimizer | Adam |
| Learning rate | 1 × 10⁻⁵ |
| Batch size | 1 |
| Iterations | 60,000 |
| EMA decay | 0.995 |
| Model channels | 128 |
| Input size | 64 × 128 × 128 |
| Timesteps | 250 |
| Sampling conditions | Real labels from source (143), pseudo-labels from target (143) |
| Sample saving frequency | Every 1,000 iterations |
| Gradient accumulation | Every 2 steps |

#### Stage 3: Segmentation model

| Parameter | Value |
|---|---|
| Model | ASC |
| Source-domain data | 143 + 143 generated samples |
| Target-domain data | 143 |
| Optimizer | Adam |
| Learning rate | 8 × 10⁻⁴ |
| Batch size | 4, with 2 source and 2 target samples |
| Epochs | 150 |
| EMA decay | 0.99 |
| Model channels | 32 |
| Input size | 64 × 128 × 128 |

**Execution order:** Stage 1 → Stage 2 → Stage 3 → Stage 2 → Stage 3.

## 5. Common Hyperparameters

| Component | Common setting |
|---|---|
| Segmentation model | ASC |
| Segmentation optimizer | Adam |
| Segmentation batch size | 4, with 2 source and 2 target samples |
| Segmentation EMA decay | 0.99 |
| Diffusion model | U-Net with one residual block |
| Diffusion optimizer | Adam |
| Diffusion learning rate | 1 × 10⁻⁵ |
| Diffusion batch size | 1 |
| Diffusion iterations | 60,000 |
| Diffusion EMA decay | 0.995 |
| Diffusion timesteps | 250 |
| Sample saving frequency | Every 1,000 iterations |
| Gradient accumulation | Every 2 steps |

## 6. Notes for Reproducibility

- All image and mask volumes are resized to the task-specific input resolution before training.
- MMWHS segmentation uses a 128 × 128 × 128 crop following an nnUNet-like cropping strategy after resizing to 144 × 144 × 144.
- Diffusion models are trained using both source-domain labels and target-domain pseudo-labels as sampling conditions.
- BraTS20 experiments use two iterative rounds of Stage 2 and Stage 3 training.
- MMWHS CT-to-MR also uses two iterative rounds of Stage 2 and Stage 3 training.
- FeTA and MMWHS MR-to-CT use one pass of Stage 1, Stage 2 and Stage 3.

## 7. Suggested Reporting Statement

> We evaluated the proposed generative UDA framework on three representative medical image segmentation benchmarks, including cross-modality cardiac segmentation on MMWHS, cross-modality brain tumor segmentation on BraTS20, and cross-center fetal brain segmentation on Atlas-to-FeTA. Each task followed a three-stage pipeline consisting of initial UDA segmentation, conditional diffusion-based data generation, and segmentation refinement using generated samples.
