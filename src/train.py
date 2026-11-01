import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import MODELS_DIR, OUTPUTS_DIR, load_mnist, normalize 

def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        raise RuntimeError("No GPU detected! Training stopped.")

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print(f"GPU detected: {gpus[0].name}")
    print("GPU training enabled.")


def build_mlp():
    return keras.Sequential([
        layers.Input((28, 28, 1)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax"),
    ], name="mnist_mlp")


def build_cnn():
   
    augment = keras.Sequential([
        layers.RandomRotation(0.06),       
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ], name="augmentation")

    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)          # 28->14, then 14->7
        return layers.Dropout(0.25)(x)

    inputs = layers.Input((28, 28, 1))
    x = augment(inputs)
    x = conv_block(x, 32)
    x = conv_block(x, 64)
    x = layers.Flatten()(x)
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="mnist_cnn")


def plot_history(history, path):
    h = history.history
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["accuracy"], label="train")
    ax[0].plot(h["val_accuracy"], label="validation")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(h["loss"], label="train")
    ax[1].plot(h["val_loss"], label="validation")
    ax[1].set_title("Loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "mlp"], default="cnn")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--data", default=None, help="optional path to mnist.npz")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    keras.utils.set_random_seed(args.seed)
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # check the GPU 
    configure_gpu()

    # 1. Data 
    (x_train, y_train), (x_test, y_test) = load_mnist(args.data)
    x_train, x_test = normalize(x_train), normalize(x_test)
    print(f"Train: {x_train.shape}  Test: {x_test.shape}")

    # Hold out 10% of training data for validation. The test set is touched
    # ONLY once, at the very end, so its score is an honest estimate.
    n_val = len(x_train) // 10
    x_val, y_val = x_train[-n_val:], y_train[-n_val:]
    x_train, y_train = x_train[:-n_val], y_train[:-n_val]

    #  2. Model
    model = build_cnn() if args.model == "cnn" else build_mlp()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",   # labels are ints 0-9
        metrics=["accuracy"],
    )
    model.summary()

    model_path = MODELS_DIR / f"mnist_{args.model}.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=2, verbose=1),
        keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy",
                                        save_best_only=True),
    ]

    # 3. Train
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # ---- 4. Final test -----------------------------------------------------
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTEST ACCURACY: {test_acc * 100:.2f}%   (loss {test_loss:.4f})")

    model.save(model_path)
    plot_history(history, OUTPUTS_DIR / f"training_curves_{args.model}.png")
    with open(OUTPUTS_DIR / f"metrics_{args.model}.json", "w") as f:
        json.dump({"model": args.model, "test_accuracy": round(test_acc, 4),
                   "test_loss": round(test_loss, 4),
                   "epochs_run": len(history.history["loss"]),
                   "params": model.count_params()}, f, indent=2)
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
