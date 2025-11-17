import numpy as np


def generate_balanced_classes(X, y):
    print("[INFO] Balancing classes ...")

    y = np.array(y)
    X = np.array(X)

    idx_wake = np.where(y == 1)[0]
    idx_non = np.where(y == 0)[0]

    n_wake = len(idx_wake)
    n_non = len(idx_non)

    print(f"[INFO] WakeWords: {n_wake}, Non-WakeWords: {n_non}")

    # Anzahl auf die kleinere Klasse reduzieren
    n_target = min(n_wake, n_non)

    # Zufällige Auswahl
    np.random.shuffle(idx_wake)
    np.random.shuffle(idx_non)

    idx_wake_bal = idx_wake[:n_target]
    idx_non_bal = idx_non[:n_target]

    # Zusammenführen
    idx_balanced = np.concatenate([idx_wake_bal, idx_non_bal])
    np.random.shuffle(idx_balanced)

    # Balancierte Daten
    X = X[idx_balanced]
    y = y[idx_balanced]

    print(f"[INFO] Balanced dataset size: {len(y)}")
    print(f"[INFO] New class distribution: wake={sum(y == 1)}, non={sum(y == 0)}")

    return X, y
