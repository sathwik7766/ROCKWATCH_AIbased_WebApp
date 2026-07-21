"""
check_dataset.py
-----------------
Run this after copying Kaggle images into data/dataset/stable/ and
data/dataset/risk/, BEFORE running train_classifier.py.

It reports how many images are in each folder and, if one class has way
more images than the other (common with the Bijie dataset: ~2003
non-landslide vs ~770 landslide), it randomly trims the larger folder's
COPY down to roughly match -- your original files are never deleted,
only copied into a new balanced folder.

Run:  python check_dataset.py
"""

import os
import random
import shutil

DATASET_DIR = "data/dataset"
BALANCED_DIR = "data/dataset_balanced"
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def count_images(folder):
    if not os.path.isdir(folder):
        return []
    return [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS]


def main():
    stable_dir = os.path.join(DATASET_DIR, "stable")
    risk_dir = os.path.join(DATASET_DIR, "risk")

    stable_files = count_images(stable_dir)
    risk_files = count_images(risk_dir)

    print(f"stable/: {len(stable_files)} images")
    print(f"risk/:   {len(risk_files)} images")

    if len(stable_files) == 0 or len(risk_files) == 0:
        print("\nOne of the folders is empty -- add images before training.")
        return

    ratio = max(len(stable_files), len(risk_files)) / min(len(stable_files), len(risk_files))

    if ratio < 1.5:
        print("\nClasses are reasonably balanced. You can train directly on data/dataset/.")
        return

    print(f"\nClasses are imbalanced ({ratio:.1f}x). Creating a balanced copy at "
          f"{BALANCED_DIR}/ so training isn't skewed toward the larger class.")

    target_count = min(len(stable_files), len(risk_files))

    for label, files, src_dir in [("stable", stable_files, stable_dir), ("risk", risk_files, risk_dir)]:
        out_dir = os.path.join(BALANCED_DIR, label)
        os.makedirs(out_dir, exist_ok=True)
        chosen = random.sample(files, target_count) if len(files) > target_count else files
        for f in chosen:
            shutil.copy(os.path.join(src_dir, f), os.path.join(out_dir, f))
        print(f"  {label}/: copied {len(chosen)} images")

    print(f"\nDone. In train_classifier.py, change:")
    print(f'    DATA_DIR = "{DATASET_DIR}"')
    print(f"to:")
    print(f'    DATA_DIR = "{BALANCED_DIR}"')
    print("then run train_classifier.py as usual.")


if __name__ == "__main__":
    main()
