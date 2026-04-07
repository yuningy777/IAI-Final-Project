import os
import copy
import csv
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)

# =========================
# Configuration
# =========================
DATA_DIR = "data/processed_4class"
MODEL_NAME = "mobilenet_v2"
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
NUM_WORKERS = 2
OPTIMIZER_NAME = "Adam"
FREEZE_BACKBONE = True
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPERIMENT_NAME = f"{MODEL_NAME}_bs{BATCH_SIZE}_ep{EPOCHS}_lr{LR}_{OPTIMIZER_NAME.lower()}_4class"

MODEL_SAVE_PATH = f"outputs/models/{EXPERIMENT_NAME}_best.pth"
TRAIN_LOG_PATH = f"outputs/logs/{EXPERIMENT_NAME}_train_log.csv"
EXPERIMENT_SUMMARY_PATH = "outputs/logs/experiment_results.csv"
CONFUSION_MATRIX_PATH = f"outputs/figures/{EXPERIMENT_NAME}_confusion_matrix.png"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloaders():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, train_dataset.classes


def build_model(num_classes, freeze_backbone=True):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    trainable_params = model.classifier[1].parameters()

    return model, trainable_params


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    return avg_loss, acc, precision, recall, f1


def evaluate_with_metrics(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    start_time = time.time()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    total_inference_time = time.time() - start_time
    avg_inference_time = total_inference_time / len(loader.dataset)

    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    return avg_loss, acc, precision, recall, f1, all_labels, all_preds, avg_inference_time


def save_train_log_header():
    os.makedirs("outputs/logs", exist_ok=True)
    with open(TRAIN_LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment_name",
            "model_name",
            "batch_size",
            "epochs",
            "learning_rate",
            "optimizer",
            "freeze_backbone",
            "epoch",
            "train_loss",
            "train_acc",
            "train_precision",
            "train_recall",
            "train_f1",
            "val_loss",
            "val_acc",
            "val_precision",
            "val_recall",
            "val_f1"
        ])


def append_train_log(row):
    with open(TRAIN_LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def append_experiment_summary(
    experiment_name,
    model_name,
    batch_size,
    epochs,
    lr,
    optimizer_name,
    freeze_backbone,
    best_val_f1,
    test_loss,
    test_acc,
    test_precision,
    test_recall,
    test_f1,
    train_time_sec,
    model_size_mb,
    inference_time_ms,
):
    file_exists = os.path.exists(EXPERIMENT_SUMMARY_PATH)

    with open(EXPERIMENT_SUMMARY_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "experiment_name",
                "model_name",
                "batch_size",
                "epochs",
                "learning_rate",
                "optimizer",
                "freeze_backbone",
                "best_val_f1",
                "test_loss",
                "test_acc",
                "test_precision_macro",
                "test_recall_macro",
                "test_f1_macro",
                "train_time_sec",
                "model_size_mb",
                "inference_time_ms_per_image"
            ])

        writer.writerow([
            experiment_name,
            model_name,
            batch_size,
            epochs,
            lr,
            optimizer_name,
            freeze_backbone,
            best_val_f1,
            test_loss,
            test_acc,
            test_precision,
            test_recall,
            test_f1,
            train_time_sec,
            model_size_mb,
            inference_time_ms,
        ])


def save_confusion_matrix(all_labels, all_preds, class_names, save_path):
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    threshold = cm.max() / 2.0 if cm.max() > 0 else 0
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
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    set_seed(SEED)

    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    num_classes = len(class_names)

    print("Classes:", class_names)
    print("Num classes:", num_classes)
    print("Device:", DEVICE)
    print("Experiment:", EXPERIMENT_NAME)

    model, trainable_params = build_model(
        num_classes=num_classes,
        freeze_backbone=FREEZE_BACKBONE,
    )
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    if OPTIMIZER_NAME == "Adam":
        optimizer = optim.Adam(trainable_params, lr=LR)
    elif OPTIMIZER_NAME == "SGD":
        optimizer = optim.SGD(trainable_params, lr=LR, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {OPTIMIZER_NAME}")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_f1 = 0.0

    save_train_log_header()

    train_start_time = time.time()

    for epoch in range(EPOCHS):
        train_loss, train_acc, train_precision, train_recall, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer
        )

        val_loss, val_acc, val_precision, val_recall, val_f1, _, _, _ = evaluate_with_metrics(
            model, val_loader, criterion
        )

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
            f"P: {train_precision:.4f} R: {train_recall:.4f} F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
            f"P: {val_precision:.4f} R: {val_recall:.4f} F1: {val_f1:.4f}"
        )

        append_train_log([
            EXPERIMENT_NAME,
            MODEL_NAME,
            BATCH_SIZE,
            EPOCHS,
            LR,
            OPTIMIZER_NAME,
            FREEZE_BACKBONE,
            epoch + 1,
            train_loss,
            train_acc,
            train_precision,
            train_recall,
            train_f1,
            val_loss,
            val_acc,
            val_precision,
            val_recall,
            val_f1
        ])

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Best model updated at epoch {epoch+1}, Val F1 = {val_f1:.4f}")

    total_train_time = time.time() - train_start_time

    model.load_state_dict(best_model_wts)

    test_loss, test_acc, test_precision, test_recall, test_f1, all_labels, all_preds, avg_inference_time = evaluate_with_metrics(
        model, test_loader, criterion
    )

    save_confusion_matrix(all_labels, all_preds, class_names, CONFUSION_MATRIX_PATH)

    model_size_mb = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)
    inference_time_ms = avg_inference_time * 1000

    print("\n===== Final Test Result =====")
    print(f"Best Val F1: {best_val_f1:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print(f"Test Precision (macro): {test_precision:.4f}")
    print(f"Test Recall (macro): {test_recall:.4f}")
    print(f"Test F1 (macro): {test_f1:.4f}")
    print(f"Training Time: {total_train_time:.2f} sec")
    print(f"Model Size: {model_size_mb:.2f} MB")
    print(f"Inference Time per Image: {inference_time_ms:.4f} ms")

    append_experiment_summary(
        experiment_name=EXPERIMENT_NAME,
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR,
        optimizer_name=OPTIMIZER_NAME,
        freeze_backbone=FREEZE_BACKBONE,
        best_val_f1=best_val_f1,
        test_loss=test_loss,
        test_acc=test_acc,
        test_precision=test_precision,
        test_recall=test_recall,
        test_f1=test_f1,
        train_time_sec=total_train_time,
        model_size_mb=model_size_mb,
        inference_time_ms=inference_time_ms,
    )

    print(f"\nTrain log saved to: {TRAIN_LOG_PATH}")
    print(f"Experiment summary saved to: {EXPERIMENT_SUMMARY_PATH}")
    print(f"Best model saved to: {MODEL_SAVE_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()