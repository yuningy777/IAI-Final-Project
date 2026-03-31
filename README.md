# Image-Based Waste Classification Assistant

## Project Goal
This project aims to classify waste images into 6 categories:
- cardboard
- glass
- metal
- paper
- plastic
- trash

The goal is to build an image-based waste classification system using transfer learning and to analyze its performance on the TrashNet dataset.

## Dataset
We use the TrashNet dataset for waste image classification.

According to the project proposal, the main challenges of this task include:
- high visual similarity between categories
- lighting variation
- background noise
- ambiguous category boundaries
- limited dataset diversity

## Project Structure
```text
IAI-Final-Project/
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── figures/
│   ├── logs/
│   └── models/
├── scripts/
│   ├── check_and_split.py
│   ├── train.py
│   ├── evaluate.py
│   ├── train_resnet_v2.py
│   └── evaluate_resnet_v2.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Environment
```bash
pip install -r requirements.txt
```

## Run Data Preprocessing
```bash
python scripts/check_and_split.py
```
This step will:
- check corrupted images
- count images in each class
- split data into train / val / test
- save dataset statistics
- generate class distribution plot

---

## Week 1

### Week 1 Objective
In Week 1, we built the first working ResNet18 baseline and made sure the full pipeline could run successfully. The Week 1 goal was to establish a complete training and evaluation framework and obtain a first baseline result that could output accuracy.

### Week 1 Setting
Baseline model:
- Model: ResNet18
- Input size: 224x224
- Pretrained on ImageNet
- Frozen backbone
- Only final classification layer is trained
- Batch size: 32
- Epochs: 10
- Optimizer: Adam
- Learning rate: 1e-3

### Run Baseline Training
```bash
python scripts/train.py
```

### Run Evaluation
```bash
python scripts/evaluate.py
```

### Week 1 Baseline Result
- Best Validation Accuracy: 0.8117
- Test Accuracy: 0.8125 

### Week 1 Output Files
After running the scripts, the following files are generated:
- `outputs/logs/dataset_stats.csv`
- `outputs/logs/split_stats.csv`
- `outputs/figures/class_distribution.png`
- `outputs/models/resnet18_baseline.pth`

---

## Week 2

### Week 2 Upgrades
In Week 2, we upgraded ResNet18 into a more formal and reportable baseline.
Compared with Week 1, this version includes:
- standard data augmentation
- precision / recall / F1 metrics
- confusion matrix analysis
- best-model saving
- experiment logging in CSV format

These evaluation metrics are also consistent with the project proposal, which identified accuracy, precision, recall, F1 score, and confusion matrix analysis as key measures of success.

### Run Week 2 Scripts
Run training:
```bash
python scripts/train_resnet_v2.py
```

Run evaluation:
```bash
python scripts/evaluate_resnet_v2.py
```

### Controlled Experiments
We conducted several experiments to compare different training settings.

| Experiment | Batch Size | Epochs | Augmentation | Learning Rate | Optimizer | Test Accuracy | Macro F1 | Notes |
| :--- | :---: | :---: | :--- | :---: | :--- | :---: | :---: | :--- |
| Week 1 Baseline | 32 | 10 | basic | 1e-3 | Adam | 0.8125 | - | original baseline |
| Baseline Extended | 32 | 15 | basic | 1e-3 | Adam | 0.8177 | - | longer training alone improves baseline |
| Week 2 Control | 32 | 10 | standard | 1e-3 | Adam | 0.8073 | 0.8066 | augmentation alone did not outperform baseline |
| Week 2 Final | 32 | 15 | standard | 1e-3 | Adam | 0.8203 | 0.8135 | best overall result |
| Extra Comparison | 128 | 15 | standard | 1e-3 | Adam | lower | lower | slower convergence, not selected |

### Final Week 2 Baseline
We selected the following setting as the formal Week 2 baseline:
- Model: ResNet18
- Batch size: 32
- Epochs: 15
- Learning rate: 1e-3
- Optimizer: Adam
- Standard augmentation: RandomResizedCrop, RandomHorizontalFlip, ColorJitter

**Final Test Performance:**
- Test Accuracy: 0.8203
- Test Precision (macro): 0.8148
- Test Recall (macro): 0.8162
- Test F1 Score (macro): 0.8135

### Key Findings
The additional baseline experiment with 15 epochs shows that a longer training schedule alone already improves the original baseline from 0.8125 to 0.8177. Therefore, the performance gain in Week 2 should not be attributed to data augmentation alone.

Under the same 15-epoch training budget, the augmented Week 2 model achieved the best result with 0.8203 test accuracy and 0.8135 macro F1, but the improvement over the non-augmented 15-epoch baseline was modest.

This suggests that the main contribution of Week 2 is not only a small performance improvement, but also a more formal and complete baseline with richer metrics, confusion matrix analysis, best-model saving, and experiment logging.

### Confusion Matrix Analysis
The confusion matrix shows several clear misclassification patterns:
- glass ↔ metal
- glass ↔ plastic
- cardboard ↔ paper

Among all categories, paper and cardboard performed strongly, while glass remained one of the most difficult classes. This is likely because glass objects often share similar shape and reflection patterns with metal and plastic containers. These findings are consistent with the project proposal, which highlighted visual similarity, lighting variation, and ambiguous category boundaries as major challenges.

### Week 2 Output Files
- `outputs/models/resnet18_bs32_ep15_lr1e-3_adam_best.pth`
- `outputs/logs/resnet18_bs32_ep15_lr1e-3_adam_train_log.csv`
- `outputs/logs/experiment_results.csv`
- `outputs/logs/resnet18_bs32_ep15_lr1e-3_adam_classification_report.txt`
- `outputs/logs/resnet18_bs32_ep15_lr1e-3_adam_test_metrics.json`
- `outputs/figures/resnet18_bs32_ep15_lr1e-3_adam_confusion_matrix.png`

### Week 2 Summary
Week 2 established a reportable formal ResNet18 baseline with complete evaluation metrics and confusion matrix analysis. The final selected model was trained with batch size 32 for 15 epochs and achieved the best overall performance among the current experiments.