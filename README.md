# AI Image Recognition for Quality Control

A computer-vision application for classifying product images as **PASS** or
**DEFECT**. It includes a ResNet-50 training pipeline, command-line inspection,
Grad-CAM defect visualizations, JSON/PDF reporting, and a Flask dashboard.

> **Important:** This is a binary image classifier, not a validated industrial
> inspection system. Its defect codes are assigned using confidence-based
> placeholder rules; the model does not identify or localize specific defect
> types. Validate it with representative production data before relying on its
> results.

## Features

- Fine-tune an ImageNet-pretrained ResNet-50 for PASS/DEFECT classification.
- Train with image augmentation, class weighting, validation metrics, early
  stopping, and TensorBoard logging.
- Inspect an image or a folder from the command line.
- Generate Grad-CAM heatmaps for images classified as DEFECT.
- Save inspection reports as JSON and, by default, PDF.
- Upload images and review inspection results in a browser-based dashboard.

## Project layout

```text
.
├── config.yaml                 # Model, training, inference, and output settings
├── dashboard.py                # Flask upload and inspection dashboard
├── inspect_images.py            # Command-line image/folder inspection
├── train.py                     # Model training entry point
├── requirements.txt
├── data/
│   ├── raw/                     # Labeled training images
│   └── reports/                 # Reports and generated heatmaps
├── models/                      # Checkpoints and training plot
├── notebooks/                   # Exploration notebook
├── src/
│   ├── model/                   # ResNet-50 and training loop
│   ├── preprocessing/            # Dataset and image transforms
│   ├── quality_control/         # Inference and Grad-CAM
│   └── reporting/               # JSON/PDF report generation
├── templates/                   # Dashboard HTML
└── tests/                       # Pytest tests
```

Some output directories (such as `data/processed/`) are configured but are not
currently used by the training or inspection commands.

## Requirements and installation

Use Python with versions of the packages listed in `requirements.txt`. PyTorch
may need a platform-specific installation if you want CUDA/GPU support; see the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/).

Run commands from the repository root:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Then install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Prepare the dataset

Training expects images in class-named folders directly under `data/raw/` (or
the directory passed with `--data_dir`):

```text
data/raw/
├── PASS/
│   ├── product_001.jpg
│   └── product_002.png
└── DEFECT/
    ├── product_101.jpg
    └── product_102.png
```

Supported file extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.tiff`.
Place one image per file in the correct class folder. The project does not
currently require a separate validation folder; it splits the dataset into
training and validation subsets using `training.val_split` in `config.yaml`
(default: 20% validation).

The existing recommendation in `data/raw/README.md` is to provide at least 500
images per class. This is guidance, not a guarantee of model quality.

## Train the model

```bash
python train.py
```

The defaults are read from `config.yaml`. For example, override the dataset,
epoch count, batch size, and checkpoint directory:

```bash
python train.py --data_dir data/raw --epochs 30 --batch_size 16 --checkpoint_dir models/
```

Available command-line options:

| Option | Default | Description |
| --- | --- | --- |
| `--config` | `config.yaml` | YAML configuration file |
| `--data_dir` | Value from config (`data/raw`) | Dataset directory |
| `--epochs` | Value from config (`50`) | Training epoch limit |
| `--batch_size` | Value from config (`32`) | Training batch size |
| `--checkpoint_dir` | `models/` | Directory for the best checkpoint and training plot |

Training saves the best validation-F1 checkpoint to
`models/best_model.pth` and the metric plot to
`models/training_history.png` by default. TensorBoard event logs are written to
`logs/`. To view them:

```bash
tensorboard --logdir logs/
```

## Inspect images

The command-line inspector requires a trained checkpoint at
`models/best_model.pth` by default. Inspect a single image:

```bash
python inspect_images.py --image path/to/product.jpg
```

Inspect every supported image in a folder:

```bash
python inspect_images.py --folder path/to/images/
```

Use another checkpoint or configuration file, or skip report generation:

```bash
python inspect_images.py --image path/to/product.jpg --model models/best_model.pth
python inspect_images.py --folder path/to/images/ --config config.yaml --no_report
```

Each successful result includes a verdict, confidence, class probabilities,
and inspection timestamp. DEFECT predictions also include a heatmap path and
one of the configured defect codes. The code assignment is heuristic and is
not a model prediction of the defect category.

Generated reports are saved under `paths.reports_dir` in `config.yaml` (default:
`data/reports/`). JSON is always generated; PDF generation is controlled by
`reporting.report_format` (default: `pdf`). Heatmaps are written to
`data/reports/heatmaps/`.

## Run the dashboard

Start the Flask application from the repository root:

```bash
python dashboard.py
```

Open [http://localhost:5000](http://localhost:5000). Set the `PORT` environment
variable to use a different port:

```powershell
# Windows PowerShell
$env:PORT = "8000"
python dashboard.py
```

```bash
# macOS/Linux
PORT=8000 python dashboard.py
```

The dashboard accepts image uploads up to 32 MB. With a trained checkpoint at
`models/best_model.pth`, uploads are passed to the model. **If the checkpoint is
missing, the dashboard returns random demo verdicts rather than real model
predictions.** Do not use demo results for quality decisions.

## Configuration

Edit `config.yaml` to adjust the model and workflow settings. Important options
include:

- `model.architecture`, `model.input_size`, `model.dropout`, and
  `model.pretrained`.
- `training.epochs`, `training.batch_size`, `training.learning_rate`,
  `training.val_split`, and `training.early_stopping_patience`.
- `classes`: the numeric-to-class mapping (`0: PASS`, `1: DEFECT`).
- `defect_codes`: labels used by the placeholder defect-code assignment.
- `paths.data_dir`, `paths.model_dir`, `paths.log_dir`, and
  `paths.reports_dir`.
- `inference.confidence_threshold` (default: `0.85`).
- `reporting.company_name` and `reporting.report_format`.

The current inference logic labels an image DEFECT when its defect probability
meets or exceeds the confidence threshold; otherwise it labels it PASS. The
threshold is not an uncertainty/review state.

## Run tests

```bash
python -m pytest
```

## Download model and dataset

An external model and dataset download is linked here:

<https://drive.google.com/drive/folders/1W7hhse78SU1CIXHiEO34lyZQDnbFkDOB?usp=drive_link>

After downloading:

- Place `best_model.pth` in `models/` for inference and dashboard predictions.
- Place labeled `PASS/` and `DEFECT/` folders under `data/raw/` for training.

Only use downloaded model files and images from sources you trust.
