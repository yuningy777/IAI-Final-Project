import os
import shutil
import random
from collections import defaultdict
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

random.seed(42)

RAW_DIR = "data/raw/TrashNet"
PROCESSED_DIR = "data/processed"
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def collect_valid_images():
    stats = []
    bad_files = []

    for cls in CLASSES:
        class_dir = os.path.join(RAW_DIR, cls)
        if not os.path.exists(class_dir):
            print(f"[Warning] Missing class folder: {class_dir}")
            continue

        files = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        valid_count = 0

        for fname in files:
            fpath = os.path.join(class_dir, fname)
            if is_valid_image(fpath):
                valid_count += 1
            else:
                bad_files.append(fpath)

        stats.append({
            "class": cls,
            "total_files": len(files),
            "valid_files": valid_count,
            "bad_files": len(files) - valid_count
        })

    return pd.DataFrame(stats), bad_files


def remove_bad_files(bad_files):
    for f in bad_files:
        print(f"Removing bad file: {f}")
        os.remove(f)


def make_dirs():
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(PROCESSED_DIR, split, cls), exist_ok=True)


def split_dataset():
    split_stats = defaultdict(lambda: defaultdict(int))

    for cls in CLASSES:
        class_dir = os.path.join(RAW_DIR, cls)
        files = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        files = sorted(files)
        random.shuffle(files)

        n = len(files)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        n_test = n - n_train - n_val

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        for split_name, split_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            for fname in split_files:
                src = os.path.join(class_dir, fname)
                dst = os.path.join(PROCESSED_DIR, split_name, cls, fname)
                shutil.copy2(src, dst)
            split_stats[cls][split_name] = len(split_files)

    return split_stats


def save_distribution_plot(df):
    plt.figure(figsize=(8, 5))
    plt.bar(df["class"], df["valid_files"])
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    os.makedirs("outputs/figures", exist_ok=True)
    plt.savefig("outputs/figures/class_distribution.png")
    plt.close()


if __name__ == "__main__":
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    df, bad_files = collect_valid_images()
    print("\n=== Before Cleaning ===")
    print(df)

    if bad_files:
        print(f"\nFound {len(bad_files)} bad files.")
        remove_bad_files(bad_files)
    else:
        print("\nNo bad files found.")

    df_after, _ = collect_valid_images()
    print("\n=== After Cleaning ===")
    print(df_after)

    df_after.to_csv("outputs/logs/dataset_stats.csv", index=False)
    save_distribution_plot(df_after)

    make_dirs()
    split_stats = split_dataset()

    split_rows = []
    for cls in CLASSES:
        split_rows.append({
            "class": cls,
            "train": split_stats[cls]["train"],
            "val": split_stats[cls]["val"],
            "test": split_stats[cls]["test"]
        })

    split_df = pd.DataFrame(split_rows)
    split_df.to_csv("outputs/logs/split_stats.csv", index=False)

    print("\n=== Split Stats ===")
    print(split_df)
    print("\nDone.")