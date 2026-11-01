import sys
from pathlib import Path

import gradio as gr
import numpy as np
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from utils import DEFAULT_MODEL_PATH, preprocess_drawn  # noqa: E402

if not DEFAULT_MODEL_PATH.exists():
    sys.exit(f"Model not found at {DEFAULT_MODEL_PATH}. Run `python src/train.py` first.")

model = keras.models.load_model(DEFAULT_MODEL_PATH)


def _to_array(value):
    """Gradio's Sketchpad returns a dict (background/layers/composite);
    Image upload returns an array. Normalise both to a numpy array."""
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("composite")
        if value is None:
            return None
    return np.asarray(value)


def predict(value):
    img = _to_array(value)
    if img is None:
        return {}, None
    x = preprocess_drawn(img)
    if x is None:
        return {}, None
    probs = model.predict(x, verbose=0)[0]
    preview = (x[0, :, :, 0] * 255).astype("uint8")
    # enlarge preview so the pixels are visible
    preview = np.kron(preview, np.ones((8, 8), dtype="uint8"))
    return {str(i): float(p) for i, p in enumerate(probs)}, preview


with gr.Blocks(title="MNIST Digit Recognizer") as demo:
    gr.Markdown(
        "#  Handwritten Digit Recognizer\n"
        "Draw a single digit **0–9** (large and centred works best) or upload "
        "a photo. A Convolutional Neural Network trained on 60,000 MNIST "
        "images predicts it."
    )
    with gr.Row():
        with gr.Column():
            with gr.Tab("Draw"):
                sketch = gr.Sketchpad(
                    label="Draw here",
                    type="numpy",
                    canvas_size=(280, 280),
                    brush=gr.Brush(default_size=18, colors=["#000000"],
                                   color_mode="fixed"),
                )
                btn_draw = gr.Button("Predict drawing", variant="primary")
            with gr.Tab("Upload"):
                upload = gr.Image(label="Photo of a digit", type="numpy")
                btn_up = gr.Button("Predict upload", variant="primary")
        with gr.Column():
            label = gr.Label(num_top_classes=3, label="Prediction")
            seen = gr.Image(label="What the model sees (28×28 after preprocessing)",
                            height=224, width=224)

    btn_draw.click(predict, sketch, [label, seen])
    sketch.change(predict, sketch, [label, seen])
    btn_up.click(predict, upload, [label, seen])


if __name__ == "__main__":
    demo.launch()
