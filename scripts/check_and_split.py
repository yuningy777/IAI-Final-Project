import json
import os
import random
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

random.seed(42)

RAW_DIR = "data/raw/TrashNet"
DATASET_DIR = "data/dataset_4class"
PROCESSED_DIR = "data/processed_4class"
CLASS_MAP = {
    "paper": ["paper", "cardboard"],
    "plastic": ["plastic"],
    "glass_metal": ["glass", "metal"],
    "trash": ["trash"],
}
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def reset_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def ensure_directories(clean=True):
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    if clean:
        reset_directory(DATASET_DIR)
        reset_directory(PROCESSED_DIR)
    else:
        os.makedirs(DATASET_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)

    for cls in CLASS_MAP:
        os.makedirs(os.path.join(DATASET_DIR, cls), exist_ok=True)
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(PROCESSED_DIR, split, cls), exist_ok=True)


def collect_valid_images():
    stats = []
    bad_files = []

    for merged_cls, raw_classes in CLASS_MAP.items():
        total_files = 0
        valid_files = 0

        for raw_cls in raw_classes:
            class_dir = os.path.join(RAW_DIR, raw_cls)
            if not os.path.exists(class_dir):
                print(f"[Warning] Missing class folder: {class_dir}")
                continue

            files = [f for f in os.listdir(class_dir) if f.lower().endswith(VALID_EXTENSIONS)]
            total_files += len(files)

            for fname in files:
                fpath = os.path.join(class_dir, fname)
                if is_valid_image(fpath):
                    valid_files += 1
                else:
                    bad_files.append(fpath)

        stats.append({
            "new_class": merged_cls,
            "original_classes": "+".join(raw_classes),
            "total_files": total_files,
            "valid_files": valid_files,
            "bad_files": total_files - valid_files,
        })

    return pd.DataFrame(stats), bad_files


def remove_bad_files(bad_files):
    for fpath in bad_files:
        print(f"Removing bad file: {fpath}")
        os.remove(fpath)


def merge_raw_to_4class():
    merge_stats = {}

    for merged_cls, raw_classes in CLASS_MAP.items():
        dest_dir = os.path.join(DATASET_DIR, merged_cls)
        copied_count = 0

        for raw_cls in raw_classes:
            src_dir = os.path.join(RAW_DIR, raw_cls)
            if not os.path.exists(src_dir):
                continue

            files = [f for f in os.listdir(src_dir) if f.lower().endswith(VALID_EXTENSIONS)]
            for fname in sorted(files):
                src_path = os.path.join(src_dir, fname)
                if not is_valid_image(src_path):
                    continue

                dst_fname = fname
                dst_path = os.path.join(dest_dir, dst_fname)

                if os.path.exists(dst_path):
                    dst_fname = f"{raw_cls}_{fname}"
                    dst_path = os.path.join(dest_dir, dst_fname)

                shutil.copy2(src_path, dst_path)
                copied_count += 1

        merged_files = [f for f in os.listdir(dest_dir) if f.lower().endswith(VALID_EXTENSIONS)]
        merge_stats[merged_cls] = {
            "copied": copied_count,
            "final_count": len(merged_files),
            "unique_count": len(set(merged_files)),
        }

    return merge_stats


def check_for_duplicate_filenames():
    print("\n=== Duplicate Check in dataset_4class ===")
    for merged_cls in CLASS_MAP:
        class_dir = os.path.join(DATASET_DIR, merged_cls)
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(VALID_EXTENSIONS)]
        total_count = len(files)
        unique_count = len(set(files))
        print(f"{merged_cls}: total={total_count}, unique={unique_count}")
        if total_count != unique_count:
            raise ValueError(f"Duplicate filenames found in {merged_cls}.")


def split_dataset():
    split_stats = defaultdict(lambda: defaultdict(int))
    split_files = {}

    for merged_cls in CLASS_MAP:
        class_dir = os.path.join(DATASET_DIR, merged_cls)
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(VALID_EXTENSIONS)]
        files = sorted(files)
        random.shuffle(files)

        if len(files) != len(set(files)):
            raise ValueError(f"Duplicate files detected before split in class: {merged_cls}")

        n = len(files)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        n_test = n - n_train - n_val

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        train_set = set(train_files)
        val_set = set(val_files)
        test_set = set(test_files)

        if train_set & val_set or train_set & test_set or val_set & test_set:
            raise ValueError(f"Overlap detected among train/val/test in class: {merged_cls}")

        split_files[merged_cls] = {
            "train": train_files,
            "val": val_files,
            "test": test_files,
        }

        for split_name, split_subset in [("train", train_files), ("val", val_files), ("test", test_files)]:
            for fname in split_subset:
                src = os.path.join(class_dir, fname)
                dst = os.path.join(PROCESSED_DIR, split_name, merged_cls, fname)
                shutil.copy2(src, dst)

        split_stats[merged_cls]["train"] = len(train_files)
        split_stats[merged_cls]["val"] = len(val_files)
        split_stats[merged_cls]["test"] = len(test_files)

    return split_stats, split_files


def save_distribution_plot(df):
    plt.figure(figsize=(8, 5))
    plt.bar(df["new_class"], df["valid_files"], color="tab:blue")
    plt.title("4-Class Data Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig("outputs/figures/class_distribution_4class.png")
    plt.close()


def save_split_files(split_files):
    split_path = "outputs/logs/split_files_4class.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_files, f, indent=2)
    print(f"Saved split file lists to: {split_path}")


if __name__ == "__main__":
    ensure_directories(clean=True)

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

    df_after.to_csv("outputs/logs/dataset_stats_4class.csv", index=False)
    save_distribution_plot(df_after)

    merge_stats = merge_raw_to_4class()
    print("\n=== Merge Stats ===")
    for cls, stats in merge_stats.items():
        print(
            f"{cls}: copied={stats['copied']}, "
            f"final_count={stats['final_count']}, "
            f"unique_count={stats['unique_count']}"
        )

    check_for_duplicate_filenames()

    split_stats, split_files = split_dataset()
    save_split_files(split_files)

    split_rows = []
    for merged_cls in CLASS_MAP:
        split_rows.append({
            "class": merged_cls,
            "train": split_stats[merged_cls]["train"],
            "val": split_stats[merged_cls]["val"],
            "test": split_stats[merged_cls]["test"],
        })

    split_df = pd.DataFrame(split_rows)
    split_df.to_csv("outputs/logs/split_stats_4class.csv", index=False)

    print("\n=== Split Stats ===")
    print(split_df)

    print("\n=== Sanity Check ===")
    for _, row in split_df.iterrows():
        cls = row["class"]
        split_total = row["train"] + row["val"] + row["test"]
        original_total = int(df_after.loc[df_after["new_class"] == cls, "valid_files"].iloc[0])
        print(f"{cls}: split_total={split_total}, expected={original_total}")

    print("\nDone.")