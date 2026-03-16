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

## Run baseline training
```bash
python scripts/train.py
```

## Run evaluation
```bash
python scripts/evaluate.py
```