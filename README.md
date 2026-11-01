## Introduction

This is my **first deep learning project**, built right after learning the fundamentals
of neural networks and **Convolutional Neural Networks (CNNs)**. I wanted to move from
theory (convolutions, pooling, backpropagation, regularisation) to something I could
actually train, test, and interact with.

The project recognises handwritten digits (0–9) from the MNIST dataset and comes with a
small web app where you can draw a digit and watch the model predict it live.



**What I focused on learning:**
- How a CNN is built layer by layer and why each layer is there
- How to train and evaluate a model properly (validation vs. test set)
- How regularisation (dropout, augmentation, early stopping) fights overfitting
- Why preprocessing at inference time must match the training data
- How to get a model running on my own GPU on Arch Linux with TensorFlow + CUDA 
- How to wrap a model into a web app with Gradio

This is only the beginning. My goal is to go deeper into GPU computing and
deep learning systems.


## Quick Start

```bash
# 1. Clone and enter the project
git clone https://github.com/Areeba-Kh571/mnist-digit-recognition.git
cd mnist-digit-recognition

# 2. create a virtual environment Linux 
python -m venv venv
# source venv/bin/activate    

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model 
python src/train.py # you can add argumentls to adjust like --model [CNN or MLp] and --epochs and --batch-size and --data and --seed

# 5. Evaluate it and generate figures
python src/evaluate.py

# 6. Launch the drawing app --> open http://127.0.0.1:7860
python app.py

