import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from utils.data_loader import load_config, load_training_data
from dataset.wake_word_dataset import WakeWordDataset
from model.model_registry import get_model
from train.evaluate import evaluate
from train.train import train
from train.early_stopping import EarlyStopping
from utils.audio_processing import normalize_segments
from utils.data_processing import generate_balanced_classes
from utils.get_device import get_device


def draw_confusion_matrix(trues, preds, writer=None):
    cm = confusion_matrix(trues, preds)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.colorbar(im, ax=ax)

    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='red')

    # Log to TensorBoard if writer provided
    if writer is not None:
        writer.add_figure('Confusion_Matrix', fig)

    plt.show()
    plt.close(fig)


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Saved model to {path}')


if __name__ == '__main__':
    cfg = load_config()

    # Get model architecture from config
    model_name = cfg.get('MODEL', 'cnn')
    print(f'Model architecture (from config.ini): {model_name}')

    X, y = load_training_data(cfg)

    X, y = generate_balanced_classes(X, y, wakeword_ratio=cfg['WAKEWORD_RATIO'])

    X = normalize_segments(X)
    X = np.expand_dims(X, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print('Shapes:')
    print('  X_train:', X_train.shape)
    print('  X_test:', X_test.shape)

    # Dataset + Dataloader
    train_ds = WakeWordDataset(X_train, y_train)
    test_ds = WakeWordDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128)

    device = get_device()
    print(f'Using device: {device}')

    # Model
    input_shape = X_train.shape[1:]
    num_classes = len(cfg['CLASSES'])

    print(f'\nCreating model: {model_name}')
    model = get_model(model_name, input_shape, num_classes).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = f'experiments/wakeword_model_{timestamp}'
    tensorboard_dir = f'{experiment_dir}/tensorboard'

    import os

    os.makedirs(experiment_dir, exist_ok=True)

    # TensorBoard setup
    writer = SummaryWriter(tensorboard_dir)

    # Log model graph
    sample_input = torch.randn(1, *input_shape).to(device)
    writer.add_graph(model, sample_input)

    print(f'\nExperiment directory: {experiment_dir}')
    print(f'TensorBoard logs: {tensorboard_dir}')
    print(f'Model will be saved to: {experiment_dir}/model.pt\n')

    # Early stopping setup
    early_stopping = EarlyStopping(
        patience=cfg['EARLY_STOPPING_PATIENCE'],
        min_delta=cfg['EARLY_STOPPING_MIN_DELTA'],
        mode='max',  # Monitoring accuracy (higher is better)
        verbose=True
    )

    # Train
    EPOCHS = cfg['EPOCHS']
    for epoch in range(EPOCHS):
        loss = train(model, train_loader, criterion, optimizer, device)
        acc, _, _ = evaluate(model, test_loader, device)

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', loss, epoch)
        writer.add_scalar('Accuracy/test', acc, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

        print(f'Epoch {epoch + 1}/{EPOCHS} | Loss {loss:.4f} | Test Acc {acc:.4f}')

        # Check early stopping
        if early_stopping(acc, epoch):
            print(f'\nEarly stopping triggered at epoch {epoch + 1}')
            print(
                f'Best accuracy: {early_stopping.get_best_score():.4f} at epoch {early_stopping.get_best_epoch() + 1}')
            break

    # Final evaluation
    acc, preds, trues = evaluate(model, test_loader, device)
    print(f'\nFinal Test Accuracy: {acc:.4f}')

    # Log final accuracy to TensorBoard
    writer.add_scalar('Accuracy/final', acc, 0)

    # Confusion Matrix
    draw_confusion_matrix(trues, preds, writer)

    # Save model to experiment directory
    model_path = f'{experiment_dir}/model.pt'
    save_model(model, model_path)

    # Also save config for reproducibility
    import json

    config_path = f'{experiment_dir}/config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'model_architecture': model_name,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'final_accuracy': float(acc),
            'epochs_trained': epoch + 1,
            'input_shape': list(input_shape),
            'num_classes': num_classes,
            'config': cfg
        }, f, indent=2)
    print(f'Saved config to {config_path}')

    # Close TensorBoard writer
    writer.close()
    print(f'\n{"=" * 60}')
    print(f'Training Complete!')
    print(f'{"=" * 60}')
    print(f'Experiment directory: {experiment_dir}')
    print(f'  - Model: {model_path}')
    print(f'  - TensorBoard: {tensorboard_dir}')
    print(f'  - Config: {config_path}')
    print(f'\nTo view TensorBoard: tensorboard --logdir=experiments')
    print(f'{"=" * 60}')
