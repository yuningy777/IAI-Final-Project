# Image-Based Waste Classification Assistant

## Project Goal
This project aims to classify waste images into 4 coarse categories by merging the original TrashNet classes:
- paper (paper + cardboard)
- plastic
- glass_metal (glass + metal)
- trash

The goal is to build an image-based waste classification system using transfer learning and to analyze its performance on the TrashNet dataset.

## Dataset
We use the TrashNet dataset for waste image classification.
The raw TrashNet dataset contains 6 original labels, and this project merges them into a 4-class assistant-style dataset for Week 3.

According to the project proposal, the main challenges of this task include:
- high visual similarity between categories
- lighting variation
- background noise
- ambiguous category boundaries
- limited dataset diversity

## Project Structure
```text
IAI-Final-Project/
  data/
    raw/
    dataset_4class/
    processed_4class/
  demo/
    app.py
    inference.py
    sample_images/
  outputs/
    figures/
    logs/
    models/
  scripts/
    check_and_split.py
    train_resnet_v2.py
    evaluate_resnet_v2.py
    train_mobilenetv2_v3.py
    evaluate_mobilenetv2_v3.py
    predict.py
    ...
  utils/
    transforms.py
  requirements.txt
  README.md
  .gitignore
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
- merge the original 6 labels into 4 classes
- split data into train / val / test on the new 4-class dataset
- save dataset statistics and a fixed split file list
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
- glass <-> metal
- glass <-> plastic
- cardboard <-> paper

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

## Week 3

### Week 3 Objective
In Week 3, we changed the original 6-class TrashNet task into a 4-class assistant-style classification task and conducted a fair comparison between ResNet18 and MobileNetV2.

The original labels were merged into:
- paper = paper + cardboard
- plastic
- glass_metal = glass + metal
- trash

The goal of Week 3 was to evaluate whether a lighter model such as MobileNetV2 could achieve competitive performance under the same experimental setting.

### Week 3 Data Preparation
We first rebuilt the dataset into a 4-class version and created a fixed train/validation/test split.

**4-class dataset statistics:**
- paper: 997
- plastic: 482
- glass_metal: 911
- trash: 137

**4-class split statistics:**
- paper: train 697 / val 149 / test 151
- plastic: train 337 / val 72 / test 73
- glass_metal: train 637 / val 136 / test 138
- trash: train 95 / val 20 / test 22

This step also verified that:
- no corrupted images remained
- no duplicate files were introduced
- split totals matched expected class totals

### Week 3 Experimental Setting
To ensure a fair comparison, both models used:
- the same 4-class dataset
- the same fixed train / val / test split
- the same image preprocessing
- the same data augmentation
- the same batch size: 32
- the same epochs: 10
- the same optimizer: Adam
- the same learning rate: 1e-3
- pretrained ImageNet weights
- frozen backbone, with only the final classifier trained

### Run Week 3 Scripts

Build the 4-class dataset and split:

```bash
python scripts/check_and_split.py
```

Train ResNet18:

```bash
python scripts/train_resnet_v2.py
```

Evaluate ResNet18:

```bash
python scripts/evaluate_resnet_v2.py
```

Train MobileNetV2:

```bash
python scripts/train_mobilenet_v2.py
```

Evaluate MobileNetV2:

```bash
python scripts/evaluate_mobilenet_v2.py
```

### Week 3 Comparative Results

| Model | Test Accuracy | Macro F1 | Training Time (s) | Model Size (MB) | Inference Time (ms/image) |
| --- | --- | --- | --- | --- | --- |
| ResNet18 | 0.8750 | 0.8213 | 571.07 | 42.72 | 51.45 |
| MobileNetV2 | 0.8880 | 0.8161 | 531.50 | 8.74 | 49.96 |

### Week 3 Confusion Matrix Analysis
Both models performed strongly on the larger classes, especially paper and glass_metal.

For ResNet18:
- paper achieved the best class-wise performance
- trash remained the most difficult class
- plastic was sometimes confused with glass_metal

For MobileNetV2:
- paper and glass_metal were also classified very well
- plastic still showed confusion with glass_metal
- trash performance was weaker than ResNet18

This suggests that both models learned the major patterns well, but minority-class recognition remains challenging, especially for trash.

### Key Findings
Week 3 produced three main findings:

1. MobileNetV2 achieved slightly higher overall test accuracy:
   - MobileNetV2: 0.8880
   - ResNet18: 0.8750

2. ResNet18 achieved slightly better macro F1:
   - ResNet18: 0.8213
   - MobileNetV2: 0.8161

3. MobileNetV2 was much more lightweight:
   - significantly smaller model size
   - slightly faster training
   - slightly faster inference

Therefore, MobileNetV2 appears more suitable for an assistant-style deployment scenario where efficiency is important, while ResNet18 provides slightly more balanced class-wise performance.

### Week 3 Output Files
- `outputs/logs/dataset_stats_4class.csv`
- `outputs/logs/split_stats_4class.csv`
- `outputs/logs/split_files_4class.json`
- `outputs/figures/class_distribution_4class.png`
- `outputs/models/resnet18_bs32_ep10_lr0.001_adam_4class_best.pth`
- `outputs/models/mobilenet_v2_bs32_ep10_lr0.001_adam_4class_best.pth`
- `outputs/logs/resnet18_bs32_ep10_lr0.001_adam_4class_train_log.csv`
- `outputs/logs/mobilenet_v2_bs32_ep10_lr0.001_adam_4class_train_log.csv`
- `outputs/logs/experiment_results.csv`
- `outputs/logs/resnet18_bs32_ep10_lr0.001_adam_4class_classification_report.txt`
- `outputs/logs/mobilenet_v2_bs32_ep10_lr0.001_adam_4class_classification_report.txt`
- `outputs/logs/resnet18_bs32_ep10_lr0.001_adam_4class_test_metrics.json`
- `outputs/logs/mobilenet_v2_bs32_ep10_lr0.001_adam_4class_test_metrics.json`
- `outputs/figures/resnet18_bs32_ep10_lr0.001_adam_4class_confusion_matrix.png`
- `outputs/figures/mobilenet_v2_bs32_ep10_lr0.001_adam_4class_confusion_matrix.png`

### Week 3 Summary
Week 3 shifted the project from the original 6-class setting to a more practical 4-class assistant-style waste classification task. Under the same experimental setup, MobileNetV2 achieved slightly higher accuracy, while ResNet18 achieved slightly better macro F1. Since MobileNetV2 is much smaller and slightly faster, it is currently the better candidate for lightweight assistant deployment.
---

## Week 4

### Week 4 Objective
In Week 4, our goal was not simply to train another model, but to study which training strategies are more effective for 4-class waste classification.

Based on the Week 3 comparison, we selected MobileNetV2 as the main model for improvement experiments because it achieved slightly higher overall accuracy while remaining lightweight and efficient.

This week, we focused on three questions:
- Does full fine-tuning work better than freezing the pretrained backbone?
- Does weighted loss help with class imbalance?
- What kinds of mistakes does the best model still make?

### Week 4 Experimental Design
We used MobileNetV2 as the common backbone and conducted the following experiments:

- E0 (Baseline): freeze backbone + standard cross-entropy loss
- E1: fine-tune all layers + standard cross-entropy loss
- E2: freeze backbone + weighted cross-entropy loss
- E3: fine-tune all layers + weighted cross-entropy loss

All experiments used:
- the same 4-class dataset
- the same fixed train / val / test split
- the same data augmentation
- the same batch size: 32
- the same image size: 224
- the same optimizer: Adam
- 10 epochs

Learning rate setting:
- 1e-3 for frozen-backbone experiments
- 1e-4 for full fine-tuning experiments

### Why Weighted Loss Was Tested
The 4-class training set is clearly imbalanced:

- glass_metal: 637
- paper: 697
- plastic: 337
- trash: 95

The trash class has far fewer samples than the other three classes, so weighted loss was tested as a possible way to improve minority-class recognition.

### Run Week 4 Scripts

#### E0: Baseline
```bash
python scripts/train_mobilenetv2_v3.py --freeze_backbone true --use_weighted_loss false --use_augmentation true --image_size 224 --epochs 10 --batch_size 32 --lr 0.001 --optimizer Adam
python scripts/evaluate_mobilenetv2_v3.py --experiment_name mobilenet_v2_img224_bs32_ep10_lr0.001_adam_freeze1_aug1_wloss0_4class --image_size 224 --top_k 10
```

#### E1: Fine-tune All Layers
```bash
python scripts/train_mobilenetv2_v3.py --freeze_backbone false --use_weighted_loss false --use_augmentation true --image_size 224 --epochs 10 --batch_size 32 --lr 0.0001 --optimizer Adam
python scripts/evaluate_mobilenetv2_v3.py --experiment_name mobilenet_v2_img224_bs32_ep10_lr0.0001_adam_freeze0_aug1_wloss0_4class --image_size 224 --top_k 10
```

#### E2: Weighted Loss
```bash
python scripts/train_mobilenetv2_v3.py --freeze_backbone true --use_weighted_loss true --use_augmentation true --image_size 224 --epochs 10 --batch_size 32 --lr 0.001 --optimizer Adam
python scripts/evaluate_mobilenetv2_v3.py --experiment_name mobilenet_v2_img224_bs32_ep10_lr0.001_adam_freeze1_aug1_wloss1_4class --image_size 224 --top_k 10
```

#### E3: Fine-tune + Weighted Loss
```bash
python scripts/train_mobilenetv2_v3.py --freeze_backbone false --use_weighted_loss true --use_augmentation true --image_size 224 --epochs 10 --batch_size 32 --lr 0.0001 --optimizer Adam
python scripts/evaluate_mobilenetv2_v3.py --experiment_name mobilenet_v2_img224_bs32_ep10_lr0.0001_adam_freeze0_aug1_wloss1_4class --image_size 224 --top_k 10
```

### Week 4 Comparative Results

| Experiment | Setting | Test Accuracy | Macro Precision | Macro Recall | Macro F1 |
| --- | --- | --- | --- | --- | --- |
| E0 | freeze + CE | 0.8802 | 0.8275 | 0.7566 | 0.7812 |
| E2 | freeze + weighted loss | 0.8542 | 0.7670 | 0.8131 | 0.7801 |
| E1 | fine-tune + CE | **0.9427** | **0.9595** | 0.8823 | **0.9129** |
| E3 | fine-tune + weighted loss | 0.9349 | 0.8765 | **0.8850** | 0.8797 |

### Minority-Class Analysis
The most difficult class throughout the experiments was still trash.

Class-wise trash performance:

| Experiment | Precision | Recall | F1 |
| --- | --- | --- | --- |
| E0 | 0.62 | 0.36 | 0.46 |
| E2 | 0.39 | 0.68 | 0.50 |
| E1 | **1.00** | 0.68 | **0.81** |
| E3 | 0.64 | **0.73** | 0.68 |

These results show that weighted loss improved trash recall, but often reduced precision. In contrast, full fine-tuning produced a much better balance and achieved the strongest trash F1.

### Top-k Error Analysis
We further analyzed the top-confidence mistakes made by the best model, E1.

Among the top 10 most confident misclassified samples:
- 6 were actually labeled as trash
- many were predicted as paper or glass_metal

Several consistent error patterns appeared:

1. trash -> paper  
   Flat white foam-like objects were often classified as paper because they share similar visual cues such as light color, large flat surfaces, and simple texture.

2. trash -> glass_metal  
   Small rigid objects, lids, or cup-like containers were often classified as glass_metal because the model relied strongly on material appearance and shape.

3. plastic -> paper  
   Some plastic samples with printed labels or light-colored packaging were predicted as paper, suggesting that the model was influenced by packaging texture and visual layout.

This shows that even the best model still makes systematic errors when categories are visually similar or semantically ambiguous.

### Key Findings
Week 4 produced four main findings:

1. Full fine-tuning was the most effective strategy.  
   Compared with the baseline E0, E1 improved test accuracy from 0.8802 to 0.9427 and improved macro F1 from 0.7812 to 0.9129.

2. Weighted loss improved minority-class recall, but introduced more false positives.  
   E2 increased macro recall and improved trash recall, but reduced overall accuracy and macro precision.

3. Adding weighted loss on top of full fine-tuning did not outperform fine-tuning alone.  
   E3 slightly increased macro recall, but reduced macro precision and macro F1 compared with E1.

4. The trash class remains the most challenging category.  
   Even under the best setting, many high-confidence mistakes were still concentrated in trash.

### Final Week 4 Model
Based on the week 4 experiments, the best overall configuration is:

- Model: MobileNetV2
- Full fine-tuning
- Batch size: 32
- Epochs: 10
- Image size: 224
- Learning rate: 1e-4
- Optimizer: Adam
- Standard data augmentation
- No weighted loss

Final Test Performance:
- Test Accuracy: 0.9427
- Test Precision (macro): 0.9595
- Test Recall (macro): 0.8823
- Test F1 Score (macro): 0.9129

### Week 4 Output Files
- outputs/logs/experiment_results_week4.csv
- outputs/logs/mobilenet_v2_img224_bs32_ep10_lr0.001_adam_freeze1_aug1_wloss0_4class_train_log.csv
- outputs/logs/mobilenet_v2_img224_bs32_ep10_lr0.001_adam_freeze1_aug1_wloss1_4class_train_log.csv
- outputs/logs/mobilenet_v2_img224_bs32_ep10_lr0.0001_adam_freeze0_aug1_wloss0_4class_train_log.csv
- outputs/logs/mobilenet_v2_img224_bs32_ep10_lr0.0001_adam_freeze0_aug1_wloss1_4class_train_log.csv
- outputs/logs/mobilenet_v2_img224_bs32_ep10_lr0.0001_adam_freeze0_aug1_wloss0_4class_classification_report.txt
- outputs/logs/mobilenet_v2_img224_bs32_ep10_lr0.0001_adam_freeze0_aug1_wloss0_4class_test_metrics.json
- outputs/logs/mobilenet_v2_img224_bs32_ep10_lr0.0001_adam_freeze0_aug1_wloss0_4class_top_errors.csv
- outputs/figures/mobilenet_v2_img224_bs32_ep10_lr0.0001_adam_freeze0_aug1_wloss0_4class_confusion_matrix_eval.png
- outputs/figures/mobilenet_v2_img224_bs32_ep10_lr0.0001_adam_freeze0_aug1_wloss0_4class_top_errors.png

### Week 4 Summary
Week 4 moved the project beyond simple model comparison and toward understanding which training strategies actually improve performance. The experiments show that full fine-tuning is the most effective method for this 4-class waste classification task, while weighted loss mainly improves minority-class recall at the cost of lower precision and overall stability. The best final model is MobileNetV2 with full fine-tuning and standard augmentation, which achieved the strongest overall result among all current experiments.

---

## Week 5

### Week 5 Objective
Week 5 focuses on locking the experimental story for the final presentation: clearly summarizing the frozen-backbone baseline versus full fine-tuning, exporting a consolidated figure set, and providing an optional inference demo. Slide files are intentionally out of scope for the repository; use the text outline and figures as the non-slide deliverables.

### Freeze Experiment (Narrative Lock)
The key Week 5 "freeze experiment" conclusion is already supported by the Week 4 E0 versus E1 comparison under standard cross-entropy:
- E0 freezes pretrained features and only trains the classifier head (lr = 1e-3).
- E1 unfreezes all layers and fine-tunes end-to-end (lr = 1e-4).

This comparison isolates the effect of unfreezing the backbone from the weighted-loss ablations.

### Regenerate Week 5 Figures (no training required)
This script reads existing CSV/JSON logs and writes presentation-ready charts to `outputs/figures/week5/`:

```bash
python scripts/week5_generate_figures.py
```

### Optional Demo (single image inference)
Requires a trained checkpoint file (for example the Week 4 E1 best weights path used in the logs):

```bash
python scripts/predict.py --image path/to/your_image.jpg
python scripts/predict.py --image path/to/your_image.jpg --json
python scripts/demo_predict.py --image path/to/your_image.jpg
```

### Gradio web demo (presentation)
Install Gradio, ensure the default checkpoint exists under `outputs/models/`, then start the local UI from the **project root**:

```bash
pip install gradio
python demo/app.py
```

Open the printed URL (typically `http://127.0.0.1:7860`). Optional: `python demo/app.py --server_port 7861` or `--checkpoint path/to/custom.pth`.

Place a few rehearsal images under `demo/sample_images/` (see `demo/sample_images/README.txt`).

### Week 5 Output Files
- `outputs/figures/week5/week5_week3_resnet_vs_mobilenet.png`
- `outputs/figures/week5/week5_week4_ablation_overview.png`
- `outputs/figures/week5/week5_freeze_vs_finetune_ce.png`
- `outputs/figures/week5/week5_weighted_loss_effect.png`
- `outputs/figures/week5/week5_train_class_counts.png`
- `outputs/figures/week5/week5_final_model_per_class_f1.png`
- `outputs/figures/week5/week5_e1_train_val_curves.png`
- `outputs/figures/week5/week5_figure_manifest.txt`
- `outputs/week5_presentation_outline.txt`
- `scripts/week5_generate_figures.py`
- `scripts/demo_predict.py`
- `scripts/predict.py`
- `demo/app.py`, `demo/inference.py`, `demo/sample_images/README.txt`
- `utils/transforms.py`

### Week 5 Summary
Week 5 packages the project for reporting: the freeze-versus-fine-tuning result is highlighted as the primary training decision, all major quantitative comparisons are exported as a single figure bundle, and a small CLI demo supports an optional live inference segment without introducing slide artifacts into the repo.