# Variational Autoencoder (VAE) with Olivetti Faces Dataset
This repository contains an implementation of a Variational Autoencoder (VAE) using PyTorch. The model is trained on the Olivetti Faces dataset from sklearn, which consists of grayscale facial images.

## Project Structure

- **main.py**              *The main script to train and test the VAE*
- **model.py**             *Defines the Variational Autoencoder architecture*
- **utils.py**             *Helper functions*
- **requirements.txt**     *Dependencies*
- **README.md**            *Explanation and usage*
- **outputs/**             *Stores generated images*
- **dataset/**             *Any dataset-related files if needed*


## Installation
To run this project, install the required dependencies:

`pip install torch torchvision numpy scikit-learn matplotlib`

## Running the Model
To train the VAE model, execute:

`python vae.py`

## Explanation
The VAE consists of:
- **Encoder**: Extracts features from input images and maps them to a latent space.
- **Reparameterization Trick**: Ensures backpropagation through a probabilistic sampling step.
- **Decoder**: Reconstructs images from the latent representation.
- **Loss Function**: Combination of Mean Squared Error (MSE) for reconstruction and KL-Divergence for latent space regularization.

## Output
The model outputs reconstructed images, comparing original and generated images.

## Example Output
The following figure demonstrates the input (top row) and corresponding reconstructed images (bottom row):
