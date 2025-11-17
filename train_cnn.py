import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from data_loader import load_config, load_training_data
from dataset.wake_word_dataset import WakeWordDataset
from model.wake_word_cnn import WakeWordCNN
from train.evaluate import evaluate
from train.train import train
from utils.audio_processing import normalize_segments
from utils.data_processing import generate_balanced_classes
from utils.get_device import get_device


def draw_confusion_matrix(trues, preds):
    cm = confusion_matrix(trues, preds)
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.colorbar()
    plt.show()


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Saved model to {path}')


if __name__ == '__main__':
    cfg = load_config()
    X, y = load_training_data(cfg)

    X, y = generate_balanced_classes(X, y)

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

    model = WakeWordCNN(input_shape, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train
    EPOCHS = cfg['EPOCHS']
    for epoch in range(EPOCHS):
        loss = train(model, train_loader, criterion, optimizer, device)
        acc, _, _ = evaluate(model, test_loader, device)
        print(f'Epoch {epoch + 1}/{EPOCHS} | Loss {loss:.4f} | Test Acc {acc:.4f}')

    # Final evaluation
    acc, preds, trues = evaluate(model, test_loader, device)
    print(f'\nFinal Test Accuracy: {acc:.4f}')

    # Confusion Matrix
    draw_confusion_matrix(trues, preds)

    # Save model
    save_model(model, 'wakeword_cnn.pt')
