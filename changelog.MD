Change log to collect all experimentation (such as dataset changes, amount of augmentation, trainingset size, ratio of
wakeword to non-wakeword etc.) for easier reporting at the end of our project.

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

**Problem:** Models were saved to `wakeword_cnn.pt` (root directory) while TensorBoard logs went to `runs/wakeword_cnn_{timestamp}/`, making it impossible to match which logs correspond to which model.

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
