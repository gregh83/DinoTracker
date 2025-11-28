# DinoTracker
App for dinosaur footprint analysis via disentangled variational autoencoder.

## Citation
This repository provides the app described in the following paper:
DOI: [to be added after acceptance]

## Repository Structure
- data/images_compressed.npz → Raw data used for training and testing (1 MB)
- data/names.npy → Names of the images (300 KB)
- data/Tone_logo_small.png → The app's logo (42 KB)
- data/tracks.xlsx → Overview of the used tracks (88 KB)
- models/model_BETA15_BIG_3k_shuffle_epoch1000.pth → beta-VAE model (2.5 MB)
- models/mu.npy → Encoding of the images (78 KB)
- src/Create_training_data.py → Creation of training data (9 KB)
- src/Training.py → Training the beta-VAE (9 KB)
- src/DinoTracker.py → The app for footprint analysis (31 KB)
- docs/Installation_guide.pdf → See below "Environment Setup" (965 KB)

## Environment
This application was tested with:
- Python 3.13.5
- PyTorch 2.9.0
- PyQt6 6.9.1
- NumPy 2.2.6
- pyqtgraph 0.13.7
- PIL 11.3.0
- h5py 3.14.0

## Environment Setup
To make this application accessible to researchers from different backgrounds, we provide a 
step‑by‑step illustrated guide for setting up the Python environment. This guide was created 
based on feedback from early testers and is available here: /docs/Installation_guide.pdf.

## Startup
`python src/DinoTracker.py`

## Feedback and Contributions

We warmly welcome feedback, feature requests, and ideas for improving this application.  
If you have suggestions or encounter any issues, please contact:

📧 gregor.hartmann@helmholtz-berlin.de

### Contributing Data
To further improve the training of our neural network, we are actively seeking additional **dinosaur footprint data**.  
- Silhouettes  
- Photographs  
- Other relevant image material  

If you have access to such data and are willing to share, your contribution would be highly valuable to expanding our dataset and enhancing the accuracy of footprint analysis.

All contributions will be acknowledged, and shared data will be used strictly for research and development purposes.

## Training Process and Hardware

The training of the network was performed on an **Apple Mac Studio** equipped with:

- M2 Ultra  
- 24‑core CPU  
- 76‑core GPU  
- 192 GB unified memory  

The code is optimized to take advantage of the large unified memory available on this system.  
For users without access to such hardware, an **alternative data loader** is provided to enable training on machines with more limited resources.

On Apple Silicon systems, the recommended backend is:
- `device = torch.device("mps")`

On other platforms, replace "mps" with "cpu" or "cuda" depending on availability:
- `device = torch.device("cpu")`   # for CPU-only systems
- `device = torch.device("cuda")` # for NVIDIA GPU systems

## Licenses of dependencies
- Python (PSF License)
- PyTorch (BSD-style)
- PyQt6 (GPL-3.0)
- NumPy (BSD)
- PyQtGraph (MIT)
- Pillow (PIL License)
- h5py (BSD)

This application is released under GPL-3.0 to comply with PyQt6 licensing. All other dependencies are permissive and compatible with GPL-3.0.
