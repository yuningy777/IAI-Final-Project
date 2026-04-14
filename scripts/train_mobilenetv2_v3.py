# To run this script, use the command line:
# python scripts/train_mobilenetv2_v3.py --freeze_backbone true/false --use_weighted_loss true/false --use_augmentation true/false --image_size 224 --epochs 10 --batch_size 32 --lr 0.001 --optimizer Adam/SGD --experiment_suffix your_suffix
# example:
# E0: python scripts/train_mobilenetv2_v3.py --freeze_backbone true --use_weighted_loss false --use_augmentation true --image_size 224 --epochs 10 --batch_size 32 --lr 0.001 --optimizer Adam
# E1: python scripts/train_mobilenetv2_v3.py --freeze_backbone false --use_weighted_loss false --use_augmentation true --image_size 224 --epochs 10 --batch_size 32 --lr 0.0001 --optimizer Adam
# E2: python scripts/train_mobilenetv2_v3.py --freeze_backbone true --use_weighted_loss true --use_augmentation true --image_size 224 --epochs 10 --batch_size 32 --lr 0.001 --optimizer Adam
# E3: python scripts/train_mobilenetv2_v3.py --freeze_backbone false --use_weighted_loss true --use_augmentation true --image_size 224 --epochs 10 --batch_size 32 --lr 0.0001 --optimizer Adam

import os
import copy
import csv
import time
import random
import argparse
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


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("true", "1", "yes", "y"):
        return True
    if v in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 for 4-class waste classification")

    parser.add_argument("--data_dir", type=str, default="data/processed_4class")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--freeze_backbone", type=str2bool, default=True)
    parser.add_argument("--use_weighted_loss", type=str2bool, default=False)
    parser.add_argument("--use_augmentation", type=str2bool, default=True)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment_suffix", type=str, default="")

    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_experiment_name(args):
    suffix = f"_{args.experiment_suffix}" if args.experiment_suffix else ""
    return (
        f"mobilenet_v2_img{args.image_size}_bs{args.batch_size}_ep{args.epochs}_"
        f"lr{args.lr}_{args.optimizer.lower()}_"
        f"freeze{int(args.freeze_backbone)}_"
        f"aug{int(args.use_augmentation)}_"
        f"wloss{int(args.use_weighted_loss)}"
        f"{suffix}_4class"
    )


def get_transforms(image_size, use_augmentation):
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    return train_transform, val_test_transform


def get_dataloaders(args):
    train_transform, val_test_transform = get_transforms(args.image_size, args.use_augmentation)

    train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "val"), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "test"), transform=val_test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def get_class_weights(train_dataset):
    num_classes = len(train_dataset.classes)
    class_counts = np.bincount(train_dataset.targets, minlength=num_classes)
    total = class_counts.sum()

    # Inverse-frequency style weighting
    class_weights = total / (num_classes * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    return class_counts, class_weights


def build_model(num_classes, freeze_backbone=True):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        trainable_params = model.classifier.parameters()
    else:
        for param in model.parameters():
            param.requires_grad = True
        trainable_params = model.parameters()

    return model, trainable_params


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

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


def evaluate_with_metrics(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    start_time = time.time()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

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


def save_train_log_header(train_log_path):
    os.makedirs(os.path.dirname(train_log_path), exist_ok=True)
    with open(train_log_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment_name",
            "batch_size",
            "epochs",
            "learning_rate",
            "optimizer",
            "freeze_backbone",
            "use_weighted_loss",
            "use_augmentation",
            "image_size",
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


def append_train_log(train_log_path, row):
    with open(train_log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def append_experiment_summary(summary_path, summary_row):
    file_exists = os.path.exists(summary_path)

    with open(summary_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "experiment_name",
                "batch_size",
                "epochs",
                "learning_rate",
                "optimizer",
                "freeze_backbone",
                "use_weighted_loss",
                "use_augmentation",
                "image_size",
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

        writer.writerow(summary_row)


def save_confusion_matrix(all_labels, all_preds, class_names, save_path, title="Confusion Matrix"):
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    experiment_name = make_experiment_name(args)

    model_save_path = f"outputs/models/{experiment_name}_best.pth"
    train_log_path = f"outputs/logs/{experiment_name}_train_log.csv"
    summary_path = "outputs/logs/experiment_results_week4.csv"
    confusion_matrix_path = f"outputs/figures/{experiment_name}_confusion_matrix.png"

    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_dataloaders(args)
    class_names = train_dataset.classes
    num_classes = len(class_names)

    class_counts, class_weights = get_class_weights(train_dataset)

    print("===== Dataset Info =====")
    print("Classes:", class_names)
    print("Class counts:", class_counts.tolist())
    print("Num train images:", len(train_dataset))
    print("Num val images:", len(val_dataset))
    print("Num test images:", len(test_dataset))

    print("\n===== Experiment Config =====")
    print("Device:", device)
    print("Experiment:", experiment_name)
    print("Freeze backbone:", args.freeze_backbone)
    print("Use weighted loss:", args.use_weighted_loss)
    print("Use augmentation:", args.use_augmentation)
    print("Image size:", args.image_size)

    model, trainable_params = build_model(
        num_classes=num_classes,
        freeze_backbone=args.freeze_backbone,
    )
    model = model.to(device)

    if args.use_weighted_loss:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("Class weights:", class_weights.tolist())
    else:
        criterion = nn.CrossEntropyLoss()

    if args.optimizer == "Adam":
        optimizer = optim.Adam(trainable_params, lr=args.lr)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(trainable_params, lr=args.lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0

    save_train_log_header(train_log_path)

    train_start_time = time.time()

    for epoch in range(args.epochs):
        train_loss, train_acc, train_precision, train_recall, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, val_precision, val_recall, val_f1, _, _, _ = evaluate_with_metrics(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
            f"P: {train_precision:.4f} R: {train_recall:.4f} F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
            f"P: {val_precision:.4f} R: {val_recall:.4f} F1: {val_f1:.4f}"
        )

        append_train_log(train_log_path, [
            experiment_name,
            args.batch_size,
            args.epochs,
            args.lr,
            args.optimizer,
            args.freeze_backbone,
            args.use_weighted_loss,
            args.use_augmentation,
            args.image_size,
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
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model updated at epoch {epoch+1}, Val F1 = {val_f1:.4f}")

    total_train_time = time.time() - train_start_time

    model.load_state_dict(best_model_wts)

    test_loss, test_acc, test_precision, test_recall, test_f1, all_labels, all_preds, avg_inference_time = evaluate_with_metrics(
        model, test_loader, criterion, device
    )

    save_confusion_matrix(
        all_labels,
        all_preds,
        class_names,
        confusion_matrix_path,
        title=f"Confusion Matrix - {experiment_name}"
    )

    model_size_mb = os.path.getsize(model_save_path) / (1024 * 1024)
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

    append_experiment_summary(summary_path, [
        experiment_name,
        args.batch_size,
        args.epochs,
        args.lr,
        args.optimizer,
        args.freeze_backbone,
        args.use_weighted_loss,
        args.use_augmentation,
        args.image_size,
        best_val_f1,
        test_loss,
        test_acc,
        test_precision,
        test_recall,
        test_f1,
        total_train_time,
        model_size_mb,
        inference_time_ms
    ])

    print(f"\nTrain log saved to: {train_log_path}")
    print(f"Experiment summary saved to: {summary_path}")
    print(f"Best model saved to: {model_save_path}")
    print(f"Confusion matrix saved to: {confusion_matrix_path}")


if __name__ == "__main__":
    main()