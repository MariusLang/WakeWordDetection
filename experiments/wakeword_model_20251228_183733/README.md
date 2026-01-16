This model was trained to test the new `crnn_with_mbconv_non_gru.py` implementation for one epoch. The Hailo compilation
worked great.

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
