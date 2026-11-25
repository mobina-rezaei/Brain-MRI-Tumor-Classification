"""
Brain Tumor MRI Classification - Windows + VS Code (Drive D)
-----------------------------------------------------------
Dataset root: D:\\BrainMRI\\Brain\\Training
Outputs:      D:\\BrainMRI\\outputs
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------
# اگر داده‌ها را کپی نکردی، این را به مسیر C تغییر بده:
# DATA_DIR = r"C:\Users\Dear User\Desktop\Brain\Training"
DATA_DIR   = r"D:\BrainMRI\Brain\Training"
OUTPUT_DIR = r"D:\BrainMRI\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED       = 123
BATCH_SIZE = 32
IMAGE_SIZE = 224
CHANNELS   = 3
EPOCHS     = 2  #30 to 2
PATIENCE   = 5
AUTOTUNE   = tf.data.AUTOTUNE

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------
# 2. Verify dataset path
# ---------------------------------------------------------------
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_DIR}\n"
        f"Make sure data is in D:\\BrainMRI\\Brain\\Training "
        f"or update DATA_DIR."
    )

print("Folders:", sorted(os.listdir(DATA_DIR)))

# ---------------------------------------------------------------
# 3. Load & split
# ---------------------------------------------------------------
train_raw = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
)

val_raw = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
)

class_names = train_raw.class_names
num_classes = len(class_names)
print("Classes:", class_names)
print("Train batches:", len(train_raw))
print("Val batches:", len(val_raw))

# ---------------------------------------------------------------
# 4. Augmentation + normalization
# ---------------------------------------------------------------
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.05, 0.05),
], name="augmentation")

def preprocess(x, y):
    return tf.cast(x, tf.float32) / 255.0, y

train_ds = (
    train_raw
    .map(lambda x, y: (augmentation(x, training=True), y),
         num_parallel_calls=AUTOTUNE)
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)
val_ds = (
    val_raw
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

# ---------------------------------------------------------------
# 5. Sample images
# ---------------------------------------------------------------
plt.figure(figsize=(10, 10))
for images, labels in train_raw.take(1):
    for i in range(min(12, len(images))):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[int(labels[i])])
        plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sample_images.png"), dpi=100)
plt.close()
print("Saved:", os.path.join(OUTPUT_DIR, "sample_images.png"))

# ---------------------------------------------------------------
# 6. Model
# ---------------------------------------------------------------
def build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
                num_classes=num_classes,
                base_filters=32,
                dropout=0.5):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(base_filters, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(base_filters * 2, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(base_filters * 4, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(base_filters * 4, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(dropout),
        layers.BatchNormalization(),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model

model = build_model()
model.summary()

# ---------------------------------------------------------------
# 7. Train
# ---------------------------------------------------------------
cb = [
    callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                            restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=2, verbose=1),
    callbacks.ModelCheckpoint(
        filepath=os.path.join(OUTPUT_DIR, "best_model.keras"),
        monitor="val_accuracy", save_best_only=True, verbose=1),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=cb,
    verbose=1,
)

# ---------------------------------------------------------------
# 8. Plot training curves
# ---------------------------------------------------------------
acc, val_acc   = history.history["accuracy"], history.history["val_accuracy"]
loss, val_loss = history.history["loss"], history.history["val_loss"]
epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, "o-", label="Train")
plt.plot(epochs, val_acc, "s-", label="Val")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, "o-", label="Train")
plt.plot(epochs, val_loss, "s-", label="Val")
plt.title("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=100)
plt.close()
print("Saved:", os.path.join(OUTPUT_DIR, "training_curves.png"))

# ---------------------------------------------------------------
# 9. Save final model
# ---------------------------------------------------------------
final_path = os.path.join(OUTPUT_DIR, "brain_tumor_cnn.keras")
model.save(final_path)
print("Model saved:", final_path)

print("\nDONE! Check:", OUTPUT_DIR)