# Perception Module: YOLOv8-enhanced DR-SPAAM

## Overview

This module contains the perception component for 2D LiDAR-based pedestrian detection. It is adapted from a DR-SPAAM-style 2D LiDAR pedestrian detection pipeline and extended with YOLOv8-based pseudo-label generation for course/research project use.

The main experiment uses YOLOv8 detections on JRDB camera images to generate visual pseudo labels, projects them into LiDAR supervision, and trains DR-SPAAM-style pedestrian detectors. In the project report, AP@0.5m improved from 0.417 to 0.566, about 35.8%.

## Module Structure

```text
Perception/src/
├── bin/                 # Training, pseudo-label generation, fusion, and comparison entry points
│   └── plotting/        # Analysis and visualization helpers
├── cfgs/                # JRDB training configs for Mask R-CNN, YOLOv8, and fused pseudo labels
├── dr_spaam/            # Core data handling, model, pipeline, detector, and utility code
├── tests/               # Local development and visualization tests
├── scripts/             # Reproduction scripts for the comparison experiments
├── README.md
├── requirements.txt
└── setup.py
```

## Key Features

- YOLOv8-based pseudo-label generation
- Mask R-CNN / YOLOv8 / fused pseudo-label comparison
- DR-SPAAM training on JRDB-style data
- Evaluation and detector comparison utilities

## Data and Weights

JRDB data is not included in this repository. Download JRDB separately and place or symlink it at an appropriate local path, for example `./data/JRDB`.

YOLOv8 weights such as `yolov8x.pt` are not included because model weights are large. Download them separately, for example by placing `yolov8x.pt` in this directory or by passing another path with `--model`.

The paths `./data/JRDB`, `./yolov8x.pt`, `./logs/`, and `./comparison_results` are placeholders used by scripts and configs. Adjust them for your local environment.

## Usage

### 1. Prepare JRDB data

```bash
python bin/setup_jrdb_dataset_v2.py
```

By default, the setup scripts use `./data/JRDB` as the JRDB root. Edit the script-level `_jrdb_dir` value or prepare a symlink if your dataset is stored elsewhere.

### 2. Generate YOLOv8 detections

```bash
python bin/generate_yolov8_detections.py \
    --data_dir ./data/JRDB \
    --split train \
    --model ./yolov8x.pt \
    --device cuda
```

### 3. Train DR-SPAAM

```bash
python bin/train.py --cfg cfgs/dr_spaam_jrdb_maskrcnn_pseudo.yaml
python bin/train.py --cfg cfgs/dr_spaam_jrdb_fused_pseudo.yaml
python bin/train.py --cfg cfgs/dr_spaam_jrdb_yolov8_pseudo.yaml
```

### 4. Compare detectors

```bash
python bin/compare_detectors.py \
    --data_dir ./data/JRDB \
    --split train \
    --output_dir ./comparison_results
```

You can also run the three training configs in sequence:

```bash
bash scripts/run_comparison_experiments.sh
```

## Notes

This directory is an archived perception module for the project. Datasets, model weights, checkpoints, and full training logs are not published with the source tree. Some setup utilities require JRDB data and ROS bag-reading dependencies that must be installed separately in the target environment.
