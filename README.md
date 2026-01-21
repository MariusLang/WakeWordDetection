# WakeWordDetection

A PyTorch-based wake word detection system optimized for deployment on Raspberry Pi with Hailo-8L AI accelerator.

> This project was developed with an emphasis on rapid iteration. During development, Claude was used as a supporting
> “coworker” to assist with refactoring and general problem-solving. In addition, several Python scripts were initially
> derived from the Jupyter notebooks of [Jakob Abeßer](https://github.com/jakobabesser) and subsequently adapted and
> extended to meet the specific requirements of this project.

## Features

- Custom wake word detection using CNN and CRNN architectures
- Audio augmentation pipeline (noise, RIR, pitch/time shift)
- Experiment tracking with TensorBoard
- Export to ONNX and Hailo HEF formats
- Real-time continuous detection on normal PCs and also on Raspberry Pis with Hailo-8L AI accelerator

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download non-wake word data
python data_preparation/download_non_wakeword.py

# Augment wake word samples (place yours in data/wakeword/ first)
python data_preparation/augment_audio.py --in data/wakeword --out data/wakeword_augmented --num_augmentations 100 \
  --natural-noise \
  --time-stretch \
  --pitch-shift \
  --gain \
  --shift \
  --reverb

# Train model
python train_cnn.py

# Test inference
python -m detection.pytorch.pytorch_inference path/to/audio.wav --model experiments/wakeword_model_TIMESTAMP/
```

For Hailo deployment, see [Model Compilation](docs/model-compilation.md) and [Deployment](docs/deployment.md).

## Project Structure

```
WakeWordDetection/
├── config.ini                 # Configuration file
├── train_cnn.py               # Main training script
├── data/                      # Training data
├── experiments/               # Training runs (auto-generated)
├── model/                     # Neural network architectures
├── detection/                 # Inference scripts
│   ├── pytorch/               # PyTorch detection (Mac/CUDA/CPU)
│   └── hailo/                 # Hailo detection (Raspberry Pi)
├── data_preparation/          # Data prep scripts
└── docs/                      # Documentation
```

## Documentation

| Guide                                          | Description                                |
|------------------------------------------------|--------------------------------------------|
| [Data Preparation](docs/data-preparation.md)   | Collecting and augmenting training data    |
| [Training](docs/training.md)                   | Model training and experiment tracking     |
| [Detection](docs/detection.md)                 | Running inference and continuous detection |
| [Model Compilation](docs/model-compilation.md) | Converting to Hailo HEF format             |
| [Deployment](docs/deployment.md)               | Raspberry Pi setup and deployment          |

## Requirements

- Python 3.12
- PyTorch 2.3+
- For Hailo deployment: Raspberry Pi with Hailo-8L accelerator
