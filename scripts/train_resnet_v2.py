import os
import copy
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# =========================
# Configuration
# =========================
DATA_DIR = "data/processed"

EXPERIMENT_NAME = "resnet18_bs32_ep10_lr1e-3_adam"

MODEL_SAVE_PATH = f"outputs/models/{EXPERIMENT_NAME}_best.pth"
TRAIN_LOG_PATH = f"outputs/logs/{EXPERIMENT_NAME}_train_log.csv"
EXPERIMENT_SUMMARY_PATH = "outputs/logs/experiment_results.csv"

NUM_CLASSES = 6
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
NUM_WORKERS = 2
OPTIMIZER_NAME = "Adam"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    return train_loader, val_loader, test_loader, train_dataset.classes


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def evaluate_with_metrics(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

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


def save_train_log_header():
    os.makedirs("outputs/logs", exist_ok=True)
    with open(TRAIN_LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment_name",
            "batch_size",
            "epochs",
            "learning_rate",
            "optimizer",
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
    batch_size,
    epochs,
    lr,
    optimizer_name,
    best_val_f1,
    test_loss,
    test_acc,
    test_precision,
    test_recall,
    test_f1
):
    file_exists = os.path.exists(EXPERIMENT_SUMMARY_PATH)

    with open(EXPERIMENT_SUMMARY_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "experiment_name",
                "batch_size",
                "epochs",
                "learning_rate",
                "optimizer",
                "best_val_f1",
                "test_loss",
                "test_acc",
                "test_precision_macro",
                "test_recall_macro",
                "test_f1_macro"
            ])

        writer.writerow([
            experiment_name,
            batch_size,
            epochs,
            lr,
            optimizer_name,
            best_val_f1,
            test_loss,
            test_acc,
            test_precision,
            test_recall,
            test_f1
        ])


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


def main():
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    print("Classes:", class_names)
    print("Device:", DEVICE)
    print("Experiment:", EXPERIMENT_NAME)

    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    if OPTIMIZER_NAME == "Adam":
        optimizer = optim.Adam(model.fc.parameters(), lr=LR)
    elif OPTIMIZER_NAME == "SGD":
        optimizer = optim.SGD(model.fc.parameters(), lr=LR, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {OPTIMIZER_NAME}")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_f1 = 0.0

    save_train_log_header()

    for epoch in range(EPOCHS):
        train_loss, train_acc, train_precision, train_recall, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer
        )

        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_with_metrics(
            model, val_loader, criterion
        )

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} P: {train_precision:.4f} R: {train_recall:.4f} F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} P: {val_precision:.4f} R: {val_recall:.4f} F1: {val_f1:.4f}"
        )

        append_train_log([
            EXPERIMENT_NAME,
            BATCH_SIZE,
            EPOCHS,
            LR,
            OPTIMIZER_NAME,
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

        # Save best model based on validation F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Best model updated at epoch {epoch+1}, Val F1 = {val_f1:.4f}")

    # Load best model for final test
    model.load_state_dict(best_model_wts)

    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_with_metrics(
        model, test_loader, criterion
    )

    print("\n===== Final Test Result =====")
    print(f"Best Val F1: {best_val_f1:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print(f"Test Precision (macro): {test_precision:.4f}")
    print(f"Test Recall (macro): {test_recall:.4f}")
    print(f"Test F1 (macro): {test_f1:.4f}")

    append_experiment_summary(
        experiment_name=EXPERIMENT_NAME,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR,
        optimizer_name=OPTIMIZER_NAME,
        best_val_f1=best_val_f1,
        test_loss=test_loss,
        test_acc=test_acc,
        test_precision=test_precision,
        test_recall=test_recall,
        test_f1=test_f1
    )

    print(f"\nTrain log saved to: {TRAIN_LOG_PATH}")
    print(f"Experiment summary saved to: {EXPERIMENT_SUMMARY_PATH}")
    print(f"Best model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()