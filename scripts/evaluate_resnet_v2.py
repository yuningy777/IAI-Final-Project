import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Configuration
EXPERIMENT_NAME = "resnet18_bs32_ep10_lr1e-3_adam"

DATA_DIR = "data/processed/test"
MODEL_PATH = f"outputs/models/{EXPERIMENT_NAME}_best.pth"
CONFUSION_MATRIX_FIG = f"outputs/figures/{EXPERIMENT_NAME}_confusion_matrix.png"
CLASSIFICATION_REPORT_TXT = f"outputs/logs/{EXPERIMENT_NAME}_classification_report.txt"
METRICS_JSON_PATH = f"outputs/logs/{EXPERIMENT_NAME}_test_metrics.json"

NUM_CLASSES = 6
BATCH_SIZE = 32
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix - {EXPERIMENT_NAME}")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
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


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    class_names = test_dataset.classes
    model = build_model()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    report_text = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0
    )

    print("===== Test Metrics =====")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1 Score:  {test_f1:.4f}")

    print("\n===== Classification Report =====")
    print(report_text)

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, CONFUSION_MATRIX_FIG)
    print(f"\nConfusion matrix saved to: {CONFUSION_MATRIX_FIG}")

    save_classification_report(report_text, CLASSIFICATION_REPORT_TXT)
    print(f"Classification report saved to: {CLASSIFICATION_REPORT_TXT}")

    metrics_dict = {
        "experiment_name": EXPERIMENT_NAME,
        "batch_size": BATCH_SIZE,
        "test_accuracy": float(test_acc),
        "test_precision_macro": float(test_precision),
        "test_recall_macro": float(test_recall),
        "test_f1_macro": float(test_f1),
        "model_path": MODEL_PATH,
        "confusion_matrix_path": CONFUSION_MATRIX_FIG,
        "classification_report_path": CLASSIFICATION_REPORT_TXT
    }
    save_metrics_json(metrics_dict, METRICS_JSON_PATH)
    print(f"Metrics json saved to: {METRICS_JSON_PATH}")


if __name__ == "__main__":
    main()