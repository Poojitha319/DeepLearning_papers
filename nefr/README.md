# Neural Radiance Field (NeRF) Implementation

This repository contains the implementation of a Neural Radiance Field (NeRF) model using PyTorch. NeRF is a novel representation that can render photorealistic images of complex 3D scenes given a set of 2D images. This implementation includes the training and rendering processes of the NeRF model.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Model Architecture](#model-architecture)
4. [Functions](#functions)
5. [Training](#training)
6. [Rendering](#rendering)
7. [Results](#results)
8. [Troubleshooting](#troubleshooting)
9. [References](#references)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Poojitha319/DeepLearning_papers/nerf
    cd nerf
    ```

2. Install the required packages:
    ```sh
    pip install torch numpy matplotlib tqdm
    ```

3. Prepare your dataset and place the training and testing data in the appropriate format in the project directory.

## Usage

1. **Training the NeRF Model:**
    ```sh
    python nerf.py
    ```

2. **Rendering Novel Views:**
    Modify the `nerf.py` to call the `test()` function with appropriate parameters after training the model.

## Model Architecture

The NeRF model consists of four main blocks:
- **Block1:** Processes the positional encodings of input 3D points.
- **Block2:** Estimates the density (sigma) at each 3D point.
- **Block3:** Processes the directional encodings of input viewing directions.
- **Block4:** Estimates the RGB color at each 3D point.

The positional and directional encodings are used to embed the input 3D coordinates and viewing directions into higher-dimensional spaces, respectively.

## Functions

### `NerfModel(nn.Module)`
Defines the NeRF model architecture with methods for positional encoding and the forward pass.

### `render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192)`
Renders pixel values by sampling points along rays and passing them through the NeRF model to get colors and densities.

### `compute_accumulated_transmittance(alphas)`
Computes the accumulated transmittance along the ray for alpha compositing.

### `test(hn, hf, dataset, chunk_size=10, img_index=0, nb_bins=192, H=400, W=400)`
Renders an image from a set of rays and saves it as a PNG file.

### `train(nerf_model, optimizer, scheduler, data_loader, device='cpu', hn=0, hf=1, nb_epochs=1e5, nb_bins=192, H=400, W=400)`
Trains the NeRF model using the provided dataset and optimizer.

## Training

The training process involves:
1. Loading the training dataset.
2. Iteratively sampling rays and their corresponding pixel values.
3. Rendering the rays using the NeRF model.
4. Computing the loss between the rendered pixel values and ground truth.
5. Backpropagating the loss and updating model parameters using the optimizer.
6. Adjusting the learning rate using a scheduler.
7. Periodically rendering test images to monitor the model's progress.

## Rendering

Rendering is performed by sampling points along rays cast through each pixel in the image. These points are passed through the NeRF model to estimate their color and density, which are then composited to form the final pixel color.

## Results

During training, rendered images are saved periodically in the `images` directory. You can visualize the progress of the model by inspecting these images.

## Troubleshooting

- **CUDA Out of Memory:** Reduce the batch size or image resolution if you encounter memory issues.
- **Slow Training:** Ensure you are using a GPU for training. You can also reduce the number of epochs or increase the chunk size to speed up training.
- **Incorrect Rendered Images:** Check if the dataset is correctly formatted and ensure the parameters (near plane, far plane, number of bins) are appropriately set.

## References

- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TQDM Documentation](https://tqdm.github.io/)
