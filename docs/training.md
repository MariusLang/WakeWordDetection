# Model Training

## Prerequisites

- Prepared training data (see [Data Preparation](data-preparation.md))
- Configuration in `config.ini` (audio params, training epochs, wakeword_ratio, etc.)

## Train the Model

```bash
python train_cnn.py
```

This loads data, generates a balanced dataset, trains with early stopping, and creates an experiment directory:
`experiments/wakeword_model_TIMESTAMP/` containing `model.pt`, `config.json`, and TensorBoard logs.

## Monitor Training

```bash
tensorboard --logdir=experiments
```

Open `http://localhost:6006` to view loss curves and accuracy metrics.

## Model Architectures

| Architecture    | Description               | Best For                 |
|-----------------|---------------------------|--------------------------|
| `cnn`           | Simple CNN (3 conv, 2 FC) | General use, fastest     |
| `crnn`          | CNN + GRU                 | Better temporal patterns |
| `crnn_temporal` | CRNN with custom GRU      | Hailo-compatible         |

## Test Your Model

```bash
python -m detection.pytorch.pytorch_inference path/to/audio.wav --model experiments/wakeword_model_TIMESTAMP/
```

## Tips

- More diverse augmentation improves generalization
- Monitor validation loss for overfitting
- Use continuous detection to collect edge cases for retraining

## Next Steps

- [Detection](detection.md) - Run inference and continuous detection
- [Model Compilation](model-compilation.md) - Deploy to Hailo accelerator
