# To run this script, use the command line:
# python scripts/evaluate_mobilenetv2_v3.py --experiment_name your_experiment
# example:
# E0: python scripts/evaluate_mobilenetv2_v3.py --experiment_name mobilenet_v2_img224_bs32_ep10_lr0.001_adam_freeze1_aug1_wloss0_4class --image_size 224 --top_k 10
# E1: python scripts/evaluate_mobilenetv2_v3.py --experiment_name mobilenet_v2_img224_bs32_ep10_lr0.0001_adam_freeze0_aug1_wloss0_4class --image_size 224 --top_k 10
# E2: python scripts/evaluate_mobilenetv2_v3.py --experiment_name mobilenet_v2_img224_bs32_ep10_lr0.001_adam_freeze1_aug1_wloss1_4class --image_size 224 --top_k 10
# E3: python scripts/evaluate_mobilenetv2_v3.py --experiment_name mobilenet_v2_img224_bs32_ep10_lr0.0001_adam_freeze0_aug1_wloss1_4class --image_size 224 --top_k 10


import os
import json
import csv
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from PIL import Image
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MobileNetV2 and export top-k error cases")

    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/processed_4class/test")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=10)

    return parser.parse_args()


def build_model(model_path, num_classes, device):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model


def plot_confusion_matrix(cm, class_names, save_path, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_classification_report(report_text, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report_text)


def save_metrics_json(metrics_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4)


def save_top_errors_csv(top_errors, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "image_path",
                "true_label",
                "pred_label",
                "pred_confidence",
                "true_class_probability"
            ]
        )
        writer.writeheader()
        writer.writerows(top_errors)


def save_top_errors_figure(top_errors, save_path):
    if len(top_errors) == 0:
        return

    n = len(top_errors)
    ncols = min(5, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = np.array(axes).reshape(-1)
    else:
        axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for idx, item in enumerate(top_errors):
        ax = axes[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(
            f'#{item["rank"]}\n'
            f'True: {item["true_label"]}\n'
            f'Pred: {item["pred_label"]}\n'
            f'Conf: {item["pred_confidence"]:.3f}',
            fontsize=10
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_name = args.experiment_name
    model_path = f"outputs/models/{experiment_name}_best.pth"

    confusion_matrix_fig = f"outputs/figures/{experiment_name}_confusion_matrix_eval.png"
    classification_report_txt = f"outputs/logs/{experiment_name}_classification_report.txt"
    metrics_json_path = f"outputs/logs/{experiment_name}_test_metrics.json"
    top_errors_csv = f"outputs/logs/{experiment_name}_top_errors.csv"
    top_errors_fig = f"outputs/figures/{experiment_name}_top_errors.png"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Test data directory not found: {args.data_dir}")

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    test_dataset = datasets.ImageFolder(args.data_dir, transform=transform)

    if len(test_dataset.classes) == 0:
        raise ValueError("No classes found in test dataset.")

    class_names = test_dataset.classes
    num_classes = len(class_names)

    print("===== Dataset Info =====")
    print("Classes:", class_names)
    print("class_to_idx:", test_dataset.class_to_idx)
    print("Num classes:", num_classes)
    print("Num test images:", len(test_dataset))

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(model_path, num_classes, device)

    all_labels = []
    all_preds = []
    top_error_candidates = []

    sample_paths = [sample[0] for sample in test_dataset.samples]
    global_idx = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            pred_confs, preds = torch.max(probs, dim=1)

            for i in range(len(labels)):
                true_label_idx = labels[i].item()
                pred_label_idx = preds[i].item()
                pred_conf = pred_confs[i].item()
                true_prob = probs[i][true_label_idx].item()

                all_labels.append(true_label_idx)
                all_preds.append(pred_label_idx)

                if true_label_idx != pred_label_idx:
                    top_error_candidates.append({
                        "image_path": sample_paths[global_idx],
                        "true_label": class_names[true_label_idx],
                        "pred_label": class_names[pred_label_idx],
                        "pred_confidence": pred_conf,
                        "true_class_probability": true_prob
                    })

                global_idx += 1

    test_acc = accuracy_score(all_labels, all_preds)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    report_text = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0
    )

    print("\n===== Test Metrics =====")
    print(f"Experiment: {experiment_name}")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1 Score:  {test_f1:.4f}")

    print("\n===== Classification Report =====")
    print(report_text)

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, confusion_matrix_fig, f"Confusion Matrix - {experiment_name}")
    print(f"Confusion matrix saved to: {confusion_matrix_fig}")

    save_classification_report(report_text, classification_report_txt)
    print(f"Classification report saved to: {classification_report_txt}")

    top_error_candidates = sorted(
        top_error_candidates,
        key=lambda x: x["pred_confidence"],
        reverse=True
    )

    top_errors = top_error_candidates[:args.top_k]
    for rank, item in enumerate(top_errors, start=1):
        item["rank"] = rank

    save_top_errors_csv(top_errors, top_errors_csv)
    print(f"Top errors csv saved to: {top_errors_csv}")

    save_top_errors_figure(top_errors, top_errors_fig)
    print(f"Top errors figure saved to: {top_errors_fig}")

    metrics_dict = {
        "experiment_name": experiment_name,
        "batch_size": args.batch_size,
        "num_test_images": len(test_dataset),
        "class_names": class_names,
        "class_to_idx": test_dataset.class_to_idx,
        "test_accuracy": float(test_acc),
        "test_precision_macro": float(test_precision),
        "test_recall_macro": float(test_recall),
        "test_f1_macro": float(test_f1),
        "model_path": model_path,
        "confusion_matrix_path": confusion_matrix_fig,
        "classification_report_path": classification_report_txt,
        "top_errors_csv": top_errors_csv,
        "top_errors_figure": top_errors_fig
    }
    save_metrics_json(metrics_dict, metrics_json_path)
    print(f"Metrics json saved to: {metrics_json_path}")


if __name__ == "__main__":
    main()