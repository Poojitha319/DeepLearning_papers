# Adversarial Feature Learning with MNIST

This project implements a neural network architecture for adversarial feature learning using the MNIST dataset. The architecture includes a Generator (G), an Encoder (E), and a Discriminator (D) network. The networks are trained to generate, encode, and discriminate between real and generated images.
Source of this paper: https://arxiv.org/abs/1605.09782

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Training](#training)
- [Results](#results)
- [License](#license)

## Introduction

Adversarial feature learning aims to learn meaningful representations of data using adversarial training. This project uses the MNIST dataset of handwritten digits to demonstrate this concept. The model consists of three main components:
- Generator (G): Generates images from latent vectors.
- Encoder (E): Encodes images into latent vectors.
- Discriminator (D): Discriminates between real and generated image-latent vector pairs.

## Installation

To run this project, you'll need to have Python 3.x installed. Follow the steps below to set up your environment:

1. Clone this repository:
    ```bash
    git clone https://github.com/Poojitha319/adversarial-feature-learning.git
    cd adversarial-feature-learning
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Open the Jupyter notebook or script where you want to run the code.
2. Ensure you have a GPU available for training (e.g., using Google Colab or a local CUDA-enabled GPU).
3. Run the training script to train the model and generate results:
    ```python
    python train.py
    ```

## Architecture

The architecture consists of three neural networks:

### Generator (G)
- Takes a 50-dimensional latent vector and outputs a 784-dimensional vector (28x28 image).
- Uses linear layers, ReLU activations, batch normalization, and a final sigmoid activation.

### Encoder (E)
- Takes a 784-dimensional input (28x28 image) and outputs a 50-dimensional latent vector.
- Uses linear layers, LeakyReLU activations, batch normalization, and a final sigmoid activation.

### Discriminator (D)
- Takes a concatenated vector of an image and a latent vector (total 834-dimensional input) and outputs a single value (real/fake probability).
- Uses linear layers, LeakyReLU activations, batch normalization, and a final sigmoid activation.

## Training

The training process involves the following steps:

1. Sample a batch of latent vectors `z` and a batch of real images `x`.
2. Compute the loss as the sum of binary cross-entropy losses for the discriminator's predictions on generated images and encoded images.
3. Update the parameters of the networks using the Adam optimizer and apply learning rate schedulers.

To train the model, run the `train.py` script:
```python
python train.py
