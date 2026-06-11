# Team CyberCorps - GLOFeagles 2026 Challenge Submission

## Overview

This project presents a deep learning pipeline for glacial lake segmentation and Glacial Lake Outburst Flood (GLOF) risk assessment using satellite imagery.

The system performs semantic segmentation of glacier-related environmental features and generates risk maps, uncertainty maps, and per-image risk reports.

---

## Team Information

**Team Name:** CyberCorps

---

## Model Architecture

The proposed solution uses **DeepLabV3+** for semantic segmentation.

### Key Features

* DeepLabV3+ segmentation architecture
* 8-channel multispectral input processing
* Combined Cross Entropy and Dice Loss
* Monte Carlo Dropout uncertainty estimation
* Automated GLOF risk assessment module

---

## Dataset

The challenge dataset contains multispectral satellite imagery and labeled glacier-related environmental classes.

### Classes

* Glacial Lake
* Snow Cover
* Cloud Cover
* Terrain Shadow
* Moraine
* Debris Cover

### Data Split

| Dataset    | Images |
| ---------- | ------ |
| Training   | 48     |
| Validation | 12     |

---

## Training Configuration

| Parameter      | Value                     |
| -------------- | ------------------------- |
| Architecture   | DeepLabV3+                |
| Input Channels | 8                         |
| Image Size     | 512 × 512                 |
| Optimizer      | AdamW                     |
| Loss Function  | Cross Entropy + Dice Loss |
| Batch Size     | 4                         |
| Early Stopping | Enabled                   |

---

## Validation Results

| Metric           | Value  |
| ---------------- | ------ |
| Best mIoU        | 0.5388 |
| Overall Accuracy | 95.22% |
| Macro F1         | 0.1998 |
| Cohen's Kappa    | 0.2062 |

---

## Generated Outputs

The trained model generated outputs for all 575 challenge images.

Generated artifacts include:

* Segmentation masks
* Visualization masks
* Uncertainty maps
* GLOF risk maps
* Per-image JSON risk reports

---

## Repository Structure

```text
checkpoints/
└── best.pt

submission_outputs/
├── predictions/
├── visualizations/
├── uncertainty/
├── risk/
└── reports/
```

---

## Running Inference

```bash
python inference.py \
  --checkpoint checkpoints/best.pt \
  --images <dataset_path> \
  --out submission_outputs \
  --mc-passes 2
```

---

## Video Demonstration

Video Link:

[Insert YouTube Link]

---

## Downloads

### Model Checkpoint

https://drive.google.com/file/d/1zYNr9ZnNXeOPr2ayxilQ5zHON-OdsNDC/view?usp=sharing

### Generated Outputs

https://drive.google.com/file/d/1DprPjc8Lg4xyteLKpKVR7V6AQqlAB7SR/view?usp=sharing

## How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python train.py
```

### Run Inference

```bash
python inference.py \
  --checkpoint checkpoints/best.pt \
  --images <input_image_folder> \
  --out submission_outputs
```

### Outputs Generated

The inference pipeline generates:

* Segmentation masks
* Visualization masks
* Uncertainty maps
* GLOF risk maps
* Per-image JSON risk reports

```
```

## Notes

* The trained model checkpoint is provided through the download link below.
* Utility and data-processing functions are implemented within `dataset.py` and the `common/` module.
* Monte Carlo Dropout is used during inference for uncertainty estimation.


## Submission Contents

* Source Code
* Trained Model Checkpoint
* Generated Outputs
* Technical Report
* README
* Video Demonstration

---

## Authors

Team CyberCorps
