from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
DEFAULT_MODEL_PATH = MODELS_DIR / "mnist_cnn.keras"


def load_mnist(npz_path=None):
    if npz_path:
        d = np.load(npz_path)
        return (d["x_train"], d["y_train"]), (d["x_test"], d["y_test"])
    import tensorflow as tf
    return tf.keras.datasets.mnist.load_data()


def normalize(x):
    return (x.astype("float32") / 255.0)[..., np.newaxis]


def preprocess_drawn(img):
  
    a = np.asarray(img, dtype="float32")

    #1  Colour / RGBA -> grayscale (an RGBA sketch keeps the strokes in alpha)
    if a.ndim == 3:
        if a.shape[2] == 4 and a[..., 3].max() > 0 and a[..., :3].std() < 1:
            a = a[..., 3]                    
        else:
            a = a[..., :3].mean(axis=2)

    if a.max() <= 0:
        return None

 
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    if border.mean() > 127:
        a = 255.0 - a

    a[a < 30] = 0                            
    rows, cols = np.nonzero(a)
    if len(rows) == 0:
        return None

    a = a[rows.min(): rows.max() + 1, cols.min(): cols.max() + 1]

    # 2) scale the longer side to 20 px
    h, w = a.shape
    scale = 20.0 / max(h, w)
    a = ndimage.zoom(a, scale, order=1)
    a = np.clip(a, 0, 255)

    # 3) paste in a 28x28 canvas, then shift so centre of mass is at the middle
    canvas = np.zeros((28, 28), dtype="float32")
    h, w = a.shape
    top, left = (28 - h) // 2, (28 - w) // 2
    canvas[top: top + h, left: left + w] = a
    cy, cx = ndimage.center_of_mass(canvas)
    canvas = ndimage.shift(canvas, (14 - cy, 14 - cx), order=1, mode="constant")

    canvas = np.clip(canvas, 0, 255) / 255.0
    return canvas.reshape(1, 28, 28, 1).astype("float32")
