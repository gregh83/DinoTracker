# DinoTracker
App for dinosaur footprint analysis via disentangled variational autoencoder.

## 📖 Citation
This repository provides the app described in the following paper:
DOI: [to be added after acceptance]

## Repository Structure
- data/images_compressed.npz → Raw data used for training and testing (1 MB)
- data/names.npy → Names of the images (300 KB)
- data/Tone_logo_small.png → The app's logo (42 KB)
- models/model_BETA15_BIG_3k_shuffle_epoch1000.pth → beta-VAE model (2.5 MB)
- models/mu.npy → Encoding of the images (78 KB)
- src/Create_training_data.py → Creation of training data (9 KB)
- src/Training.py → Training the beta-VAE (9 KB)
- src/DinoTracker_v1.0.py → The app for footprint analysis (31 KB)

## Environment
This application was tested with:
- Python 3.13.5
- PyTorch 2.9.0
- PyQt6 6.9.1
- NumPy 2.2.6
- Pyqtgraph 0.13.7
- PIL 11.3.0
- h5py 3.14.0
