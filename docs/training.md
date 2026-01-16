# Model Training

## Prerequisites

- Prepared training data (see [Data Preparation](data-preparation.md))
- Configuration in `config.ini` (audio params, training epochs, wakeword_ratio, etc.)

## Train the Model

```bash
python train_cnn.py
```

This loads data, generates a balanced dataset, trains with early stopping, and creates an experiment directory.

## Experiment Output Structure

Each training run creates a timestamped directory in `experiments/`:

```
experiments/
└── wakeword_model_YYYYMMDD_HHMMSS/
    ├── model.pt           # Trained PyTorch model weights
    ├── config.json        # Training configuration and results
    ├── tensorboard/       # TensorBoard event logs
    └── README.md          # Auto-generated experiment summary
```

**config.json** contains:

- `model_architecture` - Architecture used (cnn, crnn, crnn_temporal)
- `model_parameters` - Total trainable parameters
- `final_accuracy` - Best validation accuracy achieved
- `epochs_trained` - Number of epochs before early stopping
- `input_shape` - Model input dimensions
- `config` - Full training configuration from config.ini

## Reference Experiments

Two pre-trained experiments are included in the repository for reference (not yet fully evaluated):

| Experiment                       | Architecture  | Parameters | Epochs | Accuracy |
|----------------------------------|---------------|------------|--------|----------|
| `wakeword_model_20251212_171319` | CNN           | 137K       | 51     | 99.6%    |
| `wakeword_model_20251228_183733` | CRNN Temporal | 546K       | 30     | 99.9%    |

These can be used to test inference without training your own model.

## Visualization Notebook

The `visualize_wakeword.ipynb` notebook provides visualization and testing of the results:

- Load and play audio test files
- Display mel spectrograms
- Run inference with a selected model
- View per-segment predictions and wakeword probability ratios

Useful for understanding model behavior and debugging classification results.

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
