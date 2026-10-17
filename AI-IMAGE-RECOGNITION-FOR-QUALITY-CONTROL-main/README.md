# AI Image Recognition for Quality Control

## Overview

An end-to-end AI system that uses deep learning (CNN) to automate product quality control through image analysis, defect detection, and automated reporting.

## Project Structure

```
ai_quality_control/
├── src/
│   ├── model/          # Deep learning model (CNN)
│   ├── preprocessing/  # Image preprocessing utilities
│   ├── quality_control/# Inference & QC pipeline
│   └── reporting/      # Report generation
├── data/
│   ├── raw/            # Original product images
│   ├── processed/      # Preprocessed images
│   └── reports/        # Generated quality reports
├── notebooks/          # Jupyter exploration notebooks
├── tests/              # Unit & integration tests
├── static/             # Web dashboard assets
├── templates/          # HTML templates
├── train.py            # Model training entry point
├── inspect_images.py    # Run QC on new images
├── dashboard.py        # Launch web dashboard
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

```bash
python train.py --data_dir data/raw --epochs 50 --batch_size 32
```

### 2. Inspect Products

```bash
python inspect_images.py --image path/to/product.jpg
# or batch:
python inspect_images.py --folder path/to/images/
```

### 3. Launch Dashboard

```bash
python dashboard.py
# Open http://localhost:5000
```

## Model Architecture

- **Base**: ResNet-50 (pretrained on ImageNet)
- **Fine-tuned** for binary classification: `PASS` / `DEFECT`
- **Input**: 224×224 RGB images
- **Output**: Class label + confidence score

## Defect Categories

| Code | Description           |
| ---- | --------------------- |
| D001 | Surface scratch       |
| D002 | Dent / deformation    |
| D003 | Color mismatch        |
| D004 | Missing component     |
| D005 | Dimensional deviation |

## Reports

Reports are saved as JSON + PDF in `data/reports/`. Each report includes:

- Inspection timestamp
- Pass/Fail verdict
- Confidence score
- Defect type (if any)
- Annotated image path


## Download Large Files
Model + Dataset download link:
https://drive.google.com/drive/folders/1W7hhse78SU1CIXHiEO34lyZQDnbFkDOB?usp=drive_link

After downloading:
- Put best_model.pth in models/ folder
- Put PASS/ and DEFECT/ in data/raw/ folder
