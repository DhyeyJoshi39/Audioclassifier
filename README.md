# Audioclassifier
Audio Classification with UrbanSound8K (PyTorch + Kaggle GPU) This project builds an end‑to‑end audio classification model using the UrbanSound8K dataset and trains it on free Kaggle GPUs. The goal is to identify urban sounds such as dog barks, car horns, drilling, sirens, and more from short audio clips.
Features
End‑to‑end pipeline in a single notebook:

Dataset discovery and loading from Kaggle

Audio loading and basic preprocessing

Mel‑spectrogram feature extraction

PyTorch Dataset and DataLoader for UrbanSound8K

Convolutional Neural Network (CNN) for 10‑class sound classification

Training loop with validation and learning‑rate scheduling

Metrics and training curves (loss and accuracy)

Model saving and single‑file inference on new audio

Designed to run on:

Kaggle Notebooks with free GPU

8 GB RAM environments

Tech Stack
Python

PyTorch (model, training loop)

Librosa (audio loading, Mel‑spectrograms)

NumPy, Pandas (data handling)

Matplotlib (visualization)

Project Structure
Kaggle notebook split into clear cells:

Environment & GPU setup

Library imports

Dataset discovery in /kaggle/input

Metadata (CSV) loading and inspection

Audio file path resolution and sample loading

Mel‑spectrogram generation and plotting

UrbanSound8KDataset class and DataLoaders

CNN architecture definition

Loss, optimizer, and scheduler setup

Training & validation functions

Training loop (multiple epochs with validation)

Training/validation curves visualization

Model and metadata saving (audio_model.pth, model_metadata.json)

Inference on a random test clip with class probabilities

How It Works
The notebook automatically detects the UrbanSound8K dataset added in Kaggle’s “Add data” panel and locates the metadata CSV and audio folders.

Each audio clip is resampled, padded/trimmed to a fixed duration, and converted to a 128‑bin Mel‑spectrogram.

Spectrograms are treated as 2D images and passed to a CNN with three convolutional blocks and two fully‑connected layers.

The model is trained with cross‑entropy loss and Adam optimizer, with ReduceLROnPlateau to lower the learning rate when validation loss plateaus.

After training, the notebook saves the weights and class mappings and demonstrates prediction on unseen examples.

Results
With 15+ epochs on GPU, the model typically reaches around 75–85% validation accuracy on the 10 UrbanSound8K classes (exact numbers depend on random split and hyperparameters).

How to Run
Open the notebook on Kaggle.

Enable GPU in notebook settings.

Add an UrbanSound8K dataset from Kaggle (“Add data”).

Run all cells from top to bottom.

Use the saved audio_model.pth and model_metadata.json for inference or further experiments.
