# Data Preparation

## Overview

The training process requires:

1. **Wake word samples** - Recordings of your custom wake word in `data/wakeword/`
2. **Non-wake word samples** - Other speech/sounds in `data/non_wakeword/`

## Data Sources Used

For the reference models included in this repository:

- **Non-wake word data**: A subset of the [Google Speech Commands dataset](https://arxiv.org/abs/1804.03209), containing
  common spoken words (yes, no, up, down, numbers, etc.)
- **Wake word data**: Self-recorded samples, augmented using `data_preparation/augment_audio.py` to increase diversity
  through noise mixing, room impulse responses, and pitch/time transformations

## Step 1: Collect Wake Word Samples

Record your wake word (16kHz, mono WAV, 1-2 seconds each) and place in `data/wakeword/`. Aim for 50+ recordings with
varied speakers, environments, and speaking styles.

## Step 2: Download Non-Wake Word Data

```bash
python data_preparation/download_non_wakeword.py
```

Downloads Google Speech Commands dataset (~2GB) to `data/non_wakeword/`. Edit the script to customize which words are
downloaded.

## Step 3: Augment Wake Word Data

**Prerequisites:** Background noise files are downloaded automatically. For short noise samples, manually download
from [Freesound](https://freesound.org/) and place in `data_preparation/noises/short/`.

```bash
python data_preparation/augment_audio.py \
    --in data/wakeword \
    --out data/wakeword_augmented \
    --num_augmentations 100 \
    --normalize peak
```

**Arguments:** `-i/--in`, `-o/--out` (required), `--noise`, `--rir`, `--num_augmentations`, `--normalize` (
peak/rms/none), `--workers`, `--max_output_files`

**Augmentation techniques:** Background noise mixing, room impulse response convolution, time stretching (0.9x-1.1x),
pitch shifting (±2 semitones).

## Next Steps

Proceed to [Training](training.md).
