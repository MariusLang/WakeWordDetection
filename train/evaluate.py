import torch
from sklearn.metrics import accuracy_score


def evaluate(model, loader, device):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            out = model(X)
            pred = torch.argmax(out, dim=1).cpu().numpy()

            preds.extend(list(pred))
            trues.extend(list(y.numpy()))

    acc = accuracy_score(trues, preds)
    return acc, preds, trues
