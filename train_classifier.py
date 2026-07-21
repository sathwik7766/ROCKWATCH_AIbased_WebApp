"""
train_classifier.py
--------------------
STEP 4: Train a classifier to label a slope image as "stable" or "risk".

This version fixes "always predicts stable" (class collapse) with three
changes from the original:

  1. CLASS WEIGHTING -- if your dataset has 2003 stable vs 770 risk images,
     the model can minimize loss just by always guessing "stable" (~72%
     accuracy for zero effort). Class weights penalize that shortcut by
     making mistakes on the minority class ("risk") cost more.

  2. TRANSFER LEARNING -- instead of a tiny CNN learning from scratch on a
     few hundred images (prone to collapsing to the easy answer), this uses
     MobileNetV2 pretrained on ImageNet as a frozen feature extractor, with
     only a small trainable head on top. Far less likely to collapse, even
     on a small/imbalanced dataset.

  3. BUILT-IN SANITY CHECK -- after training, this script runs the model on
     every validation image itself and prints a per-class breakdown, so you
     immediately see if it collapsed, instead of finding out later via
     predict.py.

Expected folder structure (unchanged):
    data/dataset/
        stable/
        risk/
    (or data/dataset_balanced/ if you ran check_dataset.py)

Install first:
    pip install tensorflow pillow --break-system-packages

Run:
    python train_classifier.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------- CONFIG ----------
DATA_DIR = "data/dataset"       # change to "data/dataset_balanced" if you ran check_dataset.py
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 25
MODEL_OUT = "models/slope_classifier.h5"


def count_images_per_class(data_dir):
    """Counts images in each class folder directly -- used both to report
    the imbalance and to compute class weights."""
    counts = {}
    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            counts[class_name] = len([
                f for f in os.listdir(class_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
    return counts


def build_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    class_names = train_ds.class_names  # alphabetical: ['risk', 'stable']
    print("Classes found:", class_names)

    # MobileNetV2 expects inputs preprocessed to [-1, 1], not [0, 1] --
    # this is different from the old from-scratch version's Rescaling(1/255).
    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y)).cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y)).cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


def compute_class_weights(data_dir, class_names):
    """
    Standard inverse-frequency class weighting:
        weight[class] = total_images / (num_classes * count[class])
    A minority class with fewer images gets a higher weight, so getting it
    wrong costs the model more during training -- this is what stops it
    from just always guessing the majority class.
    """
    counts = count_images_per_class(data_dir)
    print("Image counts per class:", counts)

    total = sum(counts.values())
    num_classes = len(class_names)
    weights = {}
    for idx, name in enumerate(class_names):
        count = counts.get(name, 1)
        weights[idx] = total / (num_classes * count)

    print("Computed class weights:", {class_names[i]: round(w, 3) for i, w in weights.items()})
    ratio = max(counts.values()) / min(counts.values())
    if ratio > 1.4:
        print(f"NOTE: classes are imbalanced ({ratio:.1f}x). Class weights above "
              f"will compensate for this during training.")
    return weights


def build_model(num_classes, img_size):
    """
    Transfer learning: MobileNetV2 pretrained on ImageNet as a frozen
    feature extractor, with a small trainable classification head on top.
    This generalizes far better than a from-scratch CNN on a small dataset,
    and is much less prone to collapsing to the majority class.
    """
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

    base_model = MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # freeze -- only train the head

    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_per_class(model, val_ds, class_names):
    """
    THE SANITY CHECK. Runs the model on every validation image and prints
    a confusion-matrix-style breakdown. If you see one row all zeros (e.g.
    every "risk" image gets predicted as "stable"), that confirms collapse
    -- even if overall accuracy looks okay because the majority class
    dominates the average.
    """
    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        for true_label, pred_label in zip(labels.numpy(), pred_classes):
            confusion[true_label][pred_label] += 1

    print("\n" + "=" * 50)
    print("PER-CLASS VALIDATION BREAKDOWN (the real sanity check)")
    print("=" * 50)
    header = "true \\ predicted".ljust(18) + "".join(name.rjust(12) for name in class_names)
    print(header)
    for i, true_name in enumerate(class_names):
        row = true_name.ljust(18) + "".join(str(confusion[i][j]).rjust(12) for j in range(num_classes))
        print(row)

    print()
    collapsed = False
    for i, name in enumerate(class_names):
        total_true = confusion[i].sum()
        correct = confusion[i][i]
        if total_true == 0:
            continue
        recall = correct / total_true
        print(f"  {name}: {correct}/{total_true} correctly identified ({recall*100:.0f}% recall)")
        if recall < 0.15:
            collapsed = True

    if collapsed:
        print("\n  WARNING: at least one class has near-zero recall. The model has")
        print("  likely collapsed to always predicting the majority class. Try:")
        print("  - Re-run check_dataset.py and confirm DATA_DIR points to the balanced folder")
        print("  - Collect more images for the underrepresented class")
        print("  - Increase EPOCHS or unfreeze more of the base model")
    else:
        print("\n  Both classes have reasonable recall -- model is discriminating, not collapsed.")
    print("=" * 50 + "\n")


def main():
    train_ds, val_ds, class_names = build_datasets()
    class_weights = compute_class_weights(DATA_DIR, class_names)
    model = build_model(num_classes=len(class_names), img_size=IMG_SIZE)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    os.makedirs("models", exist_ok=True)
    model.save(MODEL_OUT)
    print(f"\nModel saved to {MODEL_OUT}")

    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    print(f"Final train accuracy: {final_train_acc:.3f}")
    print(f"Final val accuracy:   {final_val_acc:.3f}")

    # Run the real sanity check -- this is the part that tells you if it worked
    evaluate_per_class(model, val_ds, class_names)


if __name__ == "__main__":
    main()
