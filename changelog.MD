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
