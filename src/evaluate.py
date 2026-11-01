import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DEFAULT_MODEL_PATH, OUTPUTS_DIR, load_mnist, normalize  # noqa: E402


def image_grid(images, titles, colors, path, suptitle):
    n = len(images)
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.9))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, img, t, c in zip(np.ravel(axes), images, titles, colors):
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(t, fontsize=9, color=c)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--data", default=None, help="optional path to mnist.npz")
    args = p.parse_args()
    OUTPUTS_DIR.mkdir(exist_ok=True)

    model = keras.models.load_model(args.model)
    _, (x_test, y_test) = load_mnist(args.data)
    x = normalize(x_test)

    probs = model.predict(x, batch_size=512, verbose=0)
    y_pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    acc = (y_pred == y_test).mean()
    errors = np.where(y_pred != y_test)[0]
    print(f"Test accuracy: {acc * 100:.2f}%  ({len(errors)} mistakes out of {len(y_test)})")

    # Classification report
    report = classification_report(y_test, y_pred, digits=4)
    print(report)
    (OUTPUTS_DIR / "classification_report.txt").write_text(
        f"Model: {args.model}\nTest accuracy: {acc * 100:.2f}%\n\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted digit"); plt.ylabel("True digit")
    plt.title(f"Confusion matrix — accuracy {acc * 100:.2f}%")
    plt.tight_layout()
    plt.savefig(OUTPUTS_
DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # Random sample predictions
    rng = np.random.default_rng(0)
    idx = rng.choice(len(x), 24, replace=False)
    image_grid(x[idx],
               [f"pred {y_pred[i]} ({conf[i]:.0%})" for i in idx],
               ["green" if y_pred[i] == y_test[i] else "red" for i in idx],
               OUTPUTS_DIR / "sample_predictions.png", "Random test predictions")

    # Mistakes (most confident wrong answers first)
    worst = errors[np.argsort(-conf[errors])][:24]
    image_grid(x[worst],
               [f"true {y_test[i]} → {y_pred[i]}" for i in worst],
               ["red"] * len(worst),
               OUTPUTS_DIR / "misclassified.png",
               "Misclassified examples (true → predicted)")

    # Most common confusions
    off = cm.copy(); np.fill_diagonal(off, 0)
    top = np.dstack(np.unravel_index(np.argsort(-off.ravel())[:5], off.shape))[0]
    print("Most frequent confusions (true -> predicted : count):")
    for t, pr in top:
        print(f"  {t} -> {pr} : {off[t, pr]}")
    print(f"\nFigures saved in {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
