Change log to collect all experimentation (such as dataset changes, amount of augmentation, trainingset size, ratio of
wakeword to non-wakeword etc.) for easier reporting at the end of our project.

## 2025-01-10 - Refactored Detection Scripts into Modular Structure

**Changes:**

- Created a separate inference script for non hailo devices
- Refactored the inference code. Moved everything to the `/detections` directory

## 2025-12-12 - Manual Marking Feature for Model Refinement

**Investigation Goal:** Refine the wake word detection model by collecting audio samples through manual marking.

**Changes:**

- Added keyboard input ('s' key) to `continuous_wakeword.py` for manual wakeword marking during runtime
- Manual recordings save 3-second audio clips (capturing audio before the keypress) to ensure the wakeword is included
- Files saved with `manual_TIMESTAMP_####.wav` naming convention (separate from auto-detected files)

**Purpose:** Enable collection of training data from real usage scenarios where the model may have missed detections or
for gathering edge cases. These manually marked samples can be fed back into the training pipeline to improve model
accuracy and reduce false negatives.

## 2025-12-12 - Structured Experiment Tracking

**Problem:** Models were saved to `wakeword_cnn.pt` (root directory) while TensorBoard logs went to
`runs/wakeword_cnn_{timestamp}/`, making it impossible to match which logs correspond to which model.

**Changes:**

- Restructured training output to follow ML best practices
- Each training run now creates a self-contained experiment directory: `experiments/wakeword_cnn_{timestamp}/`
- Directory contains:
    - `model.pt` - trained model checkpoint
    - `tensorboard/` - TensorBoard event files
    - `config.json` - hyperparameters, final accuracy, epochs trained
- Updated `.gitignore` to exclude training artifacts

**Benefits:**

- Easy to identify and compare different training runs
- Reproducibility through saved configs
- Simplified TensorBoard comparison: `tensorboard --logdir=experiments`
- Clean repository structure

## 2025-12-12 - Flexible Model Architecture Selection

**Problem:** Training script was hardcoded to use only `WakeWordCNN`, making it difficult to experiment with different
architectures like `CRNN_with_MBConv`.

**Changes:**

- Created `model/model_registry.py` - centralized registry for all model architectures
- Added `model` parameter to `config.ini` [training] section for selecting architectures
- Fixed `CRNN_with_MBConv` to have correct default parameters (`num_classes=2`)
- Standardized model initialization interface across all architectures
- Experiment directories now include model name: `experiments/wakeword_{model}_{timestamp}/`
- `config.json` now includes model architecture name and parameter count
- Updated `inference.py` to auto-detect model architecture from experiment directories

**Usage:**

```ini
# In config.ini [training] section:
model = cnn    # Use CNN model
model = crnn   # Use CRNN model
```

## 2025-12-28 - Hailo-Compatible Temporal Convolution Model

**Problem:** The `crnn_own_gru` model (bidirectional GRU implementation) failed to compile for Hailo8L due to
unsupported `Slice` operation from `torch.flip()`, and the standard `CRNN_with_MBConv` uses bidirectional GRU which
Hailo also doesn't support well.

**Root Cause:** Hailo's compiler doesn't support:
- `torch.flip()` (creates unsupported Slice operation in ONNX)
- 1D convolutions and pooling (`Conv1d`, `AdaptiveAvgPool1d`)
- `GlobalAveragePool` / `AdaptiveAvgPool2d`
- RNN/GRU layers

**Solution:** Created `CRNN_TemporalConv` - a fully Hailo-compatible model that replaces bidirectional GRU with dilated
temporal convolutions:

**Changes:**
- Replaced `CRNN_Own_GRU` with `CRNN_TemporalConv` in `model/crnn_with_mbconv_non_gru.py`
- Uses same MBConvBlock CNN backbone as original CRNN
- Temporal modeling via dilated 2D convolutions (dilation 1, 2, 4) instead of bidirectional GRU
- Frequency dimension reduced via Conv2d instead of `mean()`
- Temporal aggregation via depthwise separable conv instead of `GlobalAveragePool`
- Updated model registry: `crnn_own_gru` renamed to `crnn_temporal`

**Model Comparison:**

| Aspect | Original GRU | Temporal Conv |
|--------|--------------|---------------|
| Parameters | 728K | 546K |
| ONNX ops | Slice (unsupported) | Conv, Add, Relu, Gemm, Flatten |
| Hailo compatible | No | Yes |

**ONNX Operations Used:** `Add`, `Conv`, `Flatten`, `Gemm`, `Identity`, `Relu` - all fully supported by Hailo.

**Usage:**
```ini
# In config.ini [training] section:
model = crnn_temporal   # Hailo-compatible temporal conv model
```

**Note:** Model is designed for fixed input shape (1, 40, 100) due to fixed-size temporal aggregation kernel.

## 2026-01-13 - Implemented Natural Noise Augmentation
**Problem:** The `generate_synthetic_noises` generates non-natural noise. 

**Root Cause:**
- Microphone doesn't generate Gaussian noise. 
- Dataset doesn't correspond to the real usage samples

**Solution:** 
Load real noise samples instead of generate synthetic noise

**Changes:**
- Added folder with real noise samples: `data_preparation/noises`
- Created function `add_natural_noise` in `data_preparation/augment_audio.py`. It loads real noise samples and mix it to source_sample
- Changed function `random_augmentation`: 
- + commented old `add_noise` function
- + added new `add_natural_noise` function