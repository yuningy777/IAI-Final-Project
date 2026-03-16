# Image-Based Waste Classification Assistant

## Project Goal
Classify waste images into 6 categories:
- cardboard
- glass
- metal
- paper
- plastic
- trash

## Dataset
TrashNet dataset

## Environment
```bash
pip install -r requirements.txt
```

## Run data preprocessing
```bash
python scripts/check_and_split.py
```
This step will:

- check corrupted images

- count images in each class

- split data into train / val / test

- save dataset statistics

- generate class distribution plot

## Run baseline training
```bash
python scripts/train.py
```
Baseline model:

- ResNet18

- input size: 224x224

- pretrained on ImageNet

- only final classification layer is trained

## Run evaluation
```bash
python scripts/evaluate.py
```

## Week 1 Baseline Result

Training result:

- Best Validation Accuracy: 0.8117

- Test Accuracy: 0.8125

## Output Files

After running the scripts, the following files are generated:

- `outputs/logs/dataset_stats.csv`

- `outputs/logs/split_stats.csv`

- `outputs/figures/class_distribution.png`

- `outputs/models/resnet18_baseline.pth`
