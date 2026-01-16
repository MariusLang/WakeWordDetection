This model was trained with non_wakeword data from the google speech command dataset. The wakeword data includes
recordings from Alisa, Matthias, and Marius. Additionally manuel refinement wav files are fed into the model. These have
been manually recorded while running the `rpi_contrinous_wakeword.py` script.

**Results:** Slightly better FP rejection rate evaluted on the Raspberry Pi. Nearly each wake word has been detected
during testing.

The following non wakewords have been used:

```python
NON_WAKE_CLASSES = [
    'yes', 'no', 'up', 'down', 'left', 'right',
    'on', 'off', 'go', 'stop', 'hi', 'hey',
    'one', 'two', 'three', 'four', 'five',
    'bed', 'cat', 'dog', 'house',
    '_background_noise_'
]
```