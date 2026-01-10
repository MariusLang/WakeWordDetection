# WakeWordDetection

A PyTorch-based wake word detection system optimized for deployment on Raspberry Pi with Hailo-8L AI accelerator. The
system uses a CNN model trained on custom wake word audio and can run real-time inference on embedded devices.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Workflow](#usage-workflow)
- [Scripts Reference](#scripts-reference)
- [Model Compilation for Hailo](#model-compilation-for-hailo)
- [Deployment](#deployment)

## Features

- Custom wake word detection using CNN architecture
- Audio augmentation pipeline with noise, RIR, and pitch/time shift
- Training with TensorBoard logging and early stopping
- Experiment tracking: Each training run creates a self-contained directory with model, config, and logs
- Export to ONNX and Hailo HEF formats for edge deployment
- Real-time continuous detection on Raspberry Pi
- Balanced class generation with configurable wakeword ratios

## Project Structure

```
WakeWordDetection/
├── config.ini                      # Configuration file for audio params, training settings
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── model_compilation.md           # Detailed Hailo compilation guide
│
├── data/                          # Training data directory
│   ├── wakeword/                  # Raw wake word audio samples
│   ├── wakeword_augmented/        # Augmented wake word samples (generated)
│   ├── non_wakeword/              # Non-wake word audio (downloaded)
│   ├── noise/                     # Background noise files for augmentation
│   └── rir/                       # Room impulse response files for augmentation
│
├── calibration_data/              # Generated calibration data for Hailo (*.npy files)
├── experiments/                   # Training experiments (auto-generated)
│   └── wakeword_cnn_TIMESTAMP/   # Each training run creates a timestamped directory
│       ├── model.pt              # Trained PyTorch model checkpoint
│       ├── config.json           # Training config and results
│       └── tensorboard/          # TensorBoard event files
├── detections/                    # Auto-detected wake word recordings (from continuous_wakeword.py)
├── recording/                     # Test recordings
│
├── model/                         # Neural network architectures
│   ├── wake_word_cnn.py               # Main CNN model definition
│   └── crnn_with_mbconv.py            # Alternative CRNN architecture
│   └── crnn_with_mbconv_non_gru.py    # Alternative CRNN architecture with custom GRU implementation for Hailo compiler
│
├── dataset/                       # Dataset loading and processing
│   ├── wake_word_dataset.py      # PyTorch Dataset class
│   ├── audio_processing.py       # Audio feature extraction
│   ├── data_loader.py            # Config loading and data utilities
│   ├── data_processing.py        # Balanced class generation
│   └── get_device.py             # GPU/CPU device selection
│
├── train/                         # Training utilities
│   ├── train.py                  # Training loop
│   ├── evaluate.py               # Evaluation functions
│   └── early_stopping.py         # Early stopping callback
│
├── utils/                         # Utility functions
│   ├── audio_processing.py       # Audio preprocessing helpers
│   ├── data_loader.py            # Config and data loading
│   ├── data_processing.py        # Data generation utilities
│   └── get_device.py             # Device detection
│
├── detection/                     # Detection and inference scripts
│   ├── base_detector.py          # Base class with shared detection logic
│   ├── pytorch/                  # PyTorch-based detection (Mac/CUDA/CPU)
│   │   ├── pytorch_detector.py   # Continuous detection
│   │   └── pytorch_inference.py  # File-based inference
│   └── hailo/                    # Hailo-based detection (Raspberry Pi)
│       ├── hailo_detector.py     # Continuous detection
│       └── hailo_inference.py    # File-based inference
│
├── data_preparation/              # Data preparation scripts
│   ├── download_non_wakeword.py  # Download Google Speech Commands dataset
│   └── augment_audio.py          # Audio augmentation pipeline
│
├── train_cnn.py                   # Main training script
├── generate_calibration_data.py  # Generate calibration data for Hailo
└── pytorch_to_hailo.py           # Convert PyTorch → ONNX
```

## Prerequisites

- Python 3.12
- PyTorch 2.3+

## Configuration

Edit `config.ini` to customize:

```ini
[audio]
sr = 16000              # Sample rate (Hz)
n_mels = 40             # Number of mel bands
n_fft = 400             # FFT window size
hop = 160               # Hop length

[segments]
segment_frames = 100    # Frames per segment (~1 second)
segments_per_file = 10  # Segments extracted per audio file

[data]
data_dir = data         # Root data directory

[classes]
class0 = non_wakeword        # Negative class directory
class1 = wakeword_augmented  # Positive class directory

[training]
epochs = 200                      # Maximum training epochs
early_stopping_patience = 15      # Early stopping patience
early_stopping_min_delta = 0.0001 # Minimum improvement threshold
wakeword_ratio = 0.25             # Ratio of wakeword to non-wakeword samples
```

## Usage Workflow

Follow these steps in order to train and deploy your wake word detector:

### 1. Data Preparation

**Step 1.1: Collect Wake Word Samples**

Record your custom wake word audio files (16kHz, mono, WAV format) and place them in:

```
data/wakeword/
```

**Step 1.2: Download Non-Wake Word Data**

```bash
cd data_preparation
python download_non_wakeword.py
```

Downloads Google Speech Commands dataset (~2GB) to `data/non_wakeword/`. You can specify the downloaded words in the
python script.

**Step 1.3: Augment Wake Word Data**

```bash
python data_preparation/augment_audio.py \
    --wakeword_in data/wakeword \
    --wakeword_out data/wakeword_augmented \
    --noise data/noise \
    --rir data/rir \
    --num_augmentations 100 \
    --normalize peak
```

Generates augmented samples with:

- Background noise mixing
- Room impulse response convolution
- Time stretching and pitch shifting
- Peak/RMS normalization

### 2. Model Training

**Step 2.1: Train the CNN Model**

```bash
python train_cnn.py
```

This script:

- Loads audio data from `data/` directory
- Generates balanced training/validation sets
- Trains WakeWordCNN model with early stopping
- Creates a timestamped experiment directory: `experiments/wakeword_cnn_TIMESTAMP/`
    - `model.pt` - Trained model checkpoint
    - `config.json` - Hyperparameters and results (accuracy, epochs)
    - `tensorboard/` - TensorBoard event files

**Step 2.2: Monitor Training**

```bash
tensorboard --logdir=experiments
```

Open `http://localhost:6006` to view training metrics.

**Step 2.3: Test Inference**

```bash
python -m detection.pytorch.pytorch_inference path/to/test_audio.wav
```

Tests the trained PyTorch model on an audio file.

### 3. Model Compilation for Hailo

See detailed guide: [model_compilation.md](model_compilation.md)

**Step 3.1: Generate Calibration Data**

```bash
python generate_calibration_data.py --num-samples 50
```

Creates `calibration_data/` with preprocessed samples for Hailo quantization.

**Step 3.2: Export to ONNX**

```bash
python pytorch_to_hailo.py \
    --pt wakeword_cnn.pt \
    --onnx wakeword.onnx \
    --shape 1 1 40 100 \
    --hw hailo8l \
    --skip_hef
```

**Step 3.3: Compile with Hailo Toolchain**

Transfer files to compilation PC and run Hailo Docker commands:

```bash
hailo parser onnx wakeword.onnx --hw-arch hailo8l
hailo optimize wakeword.har --calib-set-path calibration_data/ --hw-arch hailo8l
hailo compiler wakeword_optimized.har --hw-arch hailo8l
```

Produces `wakeword.hef` for deployment.

### 4. Deployment to Raspberry Pi

**Step 4.1: Transfer Model and Code**

```bash
# Transfer HEF model
rsync -av wakeword.hef pi@<raspberry-pi-ip>:/home/pi/WakeWordDetection/

# Sync all project files
git ls-files | rsync -av --files-from=- ./ pi@<raspberry-pi-ip>:/home/pi/WakeWordDetection/
```

**Step 4.2: Run File-based Inference**

```bash
ssh pi@<raspberry-pi-ip>
cd WakeWordDetection
python3 -m detection.hailo.hailo_inference recording/test_audio.wav
```

**Step 4.3: Run Continuous Detection**

```bash
python3 -m detection.hailo.hailo_detector \
    --hef wakeword.hef \
    --threshold 0.2 \
    --cooldown 2.0
```

Press `s` during runtime to manually mark wakewords for collecting training data.

## Scripts Reference

### Training Scripts

#### `train_cnn.py`

Main training script for the CNN wake word detector with structured experiment tracking.

**Usage:**

```bash
python train_cnn.py
```

**What it does:**

- Loads training data from directories specified in `config.ini`
- Generates balanced dataset with configurable wakeword ratio
- Trains CNN model with early stopping
- Creates experiment directory: `experiments/wakeword_cnn_TIMESTAMP/`
    - `model.pt` - Trained model weights
    - `config.json` - Training configuration and final results
    - `tensorboard/` - TensorBoard event files
- Displays confusion matrix after training

**Output structure:**

```
experiments/
└── wakeword_cnn_20251212_163045/
    ├── model.pt              # Trained model
    ├── config.json           # Hyperparameters + results
    └── tensorboard/          # TensorBoard logs
```

**Configuration:** Edit `config.ini` to change training parameters.

**Viewing results:** Use `tensorboard --logdir=experiments` to compare all training runs in one view.

### Detection Scripts

All detection and inference scripts are organized in `detection/` with separate directories for PyTorch
(Mac/CUDA/CPU) and Hailo (Raspberry Pi).

#### PyTorch Inference (`detection/pytorch/pytorch_inference.py`)

Test PyTorch model inference on audio files.

**Usage:**

```bash
python -m detection.pytorch.pytorch_inference path/to/audio.wav --model wakeword_cnn.pt
```

**Arguments:**

- `audio_file`: Path to WAV file
- `--model`: Path to model .pt file or experiment directory (default: `wakeword_cnn.pt`)
- `--threshold`: Detection threshold (default: 0.2)

**Output:**

```
Created 15 segments from file.
--- WakeWord Detection ---
Max wakeword prob: 0.956
Frames predicted as wakeword: 3/4
WAKEWORD DETECTED
```

#### Hailo Inference (`detection/hailo/hailo_inference.py`)

Run Hailo HEF model inference on Raspberry Pi.

**Usage:**

```bash
python3 -m detection.hailo.hailo_inference path/to/audio.wav --hef wakeword.hef
```

**Arguments:**

- `audio_file`: Path to WAV file
- `--hef`: Path to HEF model (default: `wakeword.hef`)
- `--threshold`: Detection threshold (default: 0.2)

#### PyTorch Continuous Detection (`detection/pytorch/pytorch_detector.py`)

Real-time continuous wake word detection using PyTorch (for Mac/CUDA/CPU).

**Usage:**

```bash
python -m detection.pytorch.pytorch_detector \
    --model wakeword_cnn.pt \
    --threshold 0.2 \
    --cooldown 2.0 \
    --detection-dir detections
```

**Arguments:**

- `--model`: Path to .pt/.pth state_dict or checkpoint (required)
- `--model-name`: Force architecture (cnn, crnn, crnn_temporal)
- `--threshold`: Detection probability threshold (default: 0.2)
- `--cooldown`: Seconds between detections (default: 2.0)
- `--no-save`: Do not save detection audio files
- `--detection-dir`: Directory for saved detections (default: `detections`)
- `--device`: Force device (cpu, mps, cuda)

#### Hailo Continuous Detection (`detection/hailo/hailo_detector.py`)

Real-time continuous wake word detection using Hailo accelerator (Raspberry Pi).

**Usage:**

```bash
python3 -m detection.hailo.hailo_detector \
    --hef wakeword.hef \
    --threshold 0.2 \
    --cooldown 2.0 \
    --detection-dir detections
```

**Arguments:**

- `--hef`: Path to HEF model (default: `wakeword.hef`)
- `--threshold`: Detection probability threshold (default: 0.2)
- `--cooldown`: Seconds between detections (default: 2.0)
- `--no-save`: Do not save detection audio files
- `--detection-dir`: Directory for saved detections (default: `detections`)

**Manual Marking (both detectors):**

While the script is running, press `s` to manually mark that a wakeword was just spoken. This saves a 3-second audio
clip (capturing audio before the keypress) for model refinement. Files are saved as:

- Auto-detected: `detection_TIMESTAMP_prob0.XXX_ratio0.XXX_####.wav`
- Manual marks: `manual_TIMESTAMP_####.wav`

Use manual marks to collect edge cases where the model missed detections, then feed them back into training to improve
accuracy.

**Microphone Configuration for Raspberry Pi:**



### Data Preparation Scripts

#### `data_preparation/download_non_wakeword.py`

Downloads Google Speech Commands dataset for negative samples.

**Usage:**

```bash
cd data_preparation
python download_non_wakeword.py
```

**What it does:**

- Downloads Speech Commands v0.02 (~2GB)
- Extracts specific classes (yes, no, up, down, numbers, etc.)
- Saves to `data/non_wakeword/`

#### `data_preparation/augment_audio.py`

Augments wake word audio with noise, RIR, and transformations.

**Usage:**

```bash
python data_preparation/augment_audio.py \
    --wakeword_in data/wakeword \
    --wakeword_out data/wakeword_augmented \
    --noise data/noise \
    --rir data/rir \
    --num_augmentations 100 \
    --normalize peak \
    --workers 4
```

**Arguments:**

- `--wakeword_in`: Input directory with raw wake word samples
- `--wakeword_out`: Output directory for augmented samples
- `--noise`: Directory with noise files
- `--rir`: Directory with room impulse response files
- `--num_augmentations`: Augmentations per input file (default: 10)
- `--normalize`: Normalization method: `peak`, `rms`, or `none` (default: `peak`)
- `--workers`: Parallel workers (default: CPU count)

**Augmentation techniques:**

- Background noise addition (various SNR levels)
- Room impulse response convolution
- Time stretching (0.9x - 1.1x)
- Pitch shifting (±2 semitones)

### Model Compilation Scripts

#### `generate_calibration_data.py`

Generates calibration data for Hailo quantization.

**Usage:**

```bash
python generate_calibration_data.py --num-samples 50
```

**Arguments:**

- `--num-samples`: Number of calibration samples (default: 50)
- `--wakeword-ratio`: Ratio of wakeword samples (default: from config)
- `--min-class-samples`: Minimum samples per class (default: 10)

**Output:** Creates `calibration_data/sample_XXXX.npy` files with preprocessed mel spectrograms.

#### `pytorch_to_hailo.py`

Converts PyTorch model to ONNX format for Hailo compilation.

**Usage:**

```bash
python pytorch_to_hailo.py \
    --pt wakeword_cnn.pt \
    --onnx wakeword.onnx \
    --shape 1 1 40 100 \
    --hw hailo8l \
    --skip_hef
```

**Arguments:**

- `--pt`: Input PyTorch model file
- `--onnx`: Output ONNX file
- `--shape`: Input shape (batch, channels, height, width)
- `--hw`: Target hardware: `hailo8`, `hailo8l` (default: `hailo8l`)
- `--skip_hef`: Only export ONNX, skip HEF compilation

## Model Compilation for Hailo

For complete step-by-step instructions on compiling your model for the Hailo-8L accelerator, see:

**[model_compilation.md](model_compilation.md)**

This guide covers:

- Generating calibration data
- Exporting PyTorch → ONNX
- Using Hailo Docker toolchain (parse, optimize, compile)
- Transferring files between Mac, compilation PC, and Raspberry Pi
- Running inference on Raspberry Pi
- Troubleshooting common issues

## Deployment

### File Sync Commands

Sync all git-tracked files to compilation PC:

```bash
git ls-files | rsync -av --files-from=- ./ marius@192.168.178.62:/home/marius/Downloads/WakeWordDetection
```

Sync to Raspberry Pi:

```bash
git ls-files | rsync -av --files-from=- ./ pi@192.168.178.194:/home/pi/WakeWordDetection
```

### Running on Raspberry Pi

1. **Test with audio file:**

```bash
python3 -m detection.hailo.hailo_inference recording/test.wav
```

2. **Continuous detection:**

```bash
python3 -m detection.hailo.hailo_detector --threshold 0.2
```

## Performance

TODO

## Further Reading

TODO