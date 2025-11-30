import numpy as np


def generate_balanced_classes(X, y, wakeword_ratio=0.5):
    """
    Balance classes according to a specified ratio.

    Args:
        X: Feature array
        y: Labels array
        wakeword_ratio: Ratio of wakeword samples in the final dataset (0.0 to 1.0)
                       E.g., 0.5 = 50% wakewords, 50% non-wakewords
                             0.3 = 30% wakewords, 70% non-wakewords

    Returns:
        Balanced X and y arrays
    """
    print(f"[INFO] Balancing classes with wakeword ratio: {wakeword_ratio:.2f}")

    y = np.array(y)
    X = np.array(X)

    idx_wake = np.where(y == 1)[0]
    idx_non = np.where(y == 0)[0]

    n_wake = len(idx_wake)
    n_non = len(idx_non)

    print(f"[INFO] Original - WakeWords: {n_wake}, Non-WakeWords: {n_non}")

    # Calculate target counts based on ratio
    # We want: n_wake_target / (n_wake_target + n_non_target) = wakeword_ratio
    # We use the smaller class as reference to avoid upsampling

    if wakeword_ratio == 0.5:
        # Special case: 50/50 split
        n_target = min(n_wake, n_non)
        n_wake_target = n_target
        n_non_target = n_target
    elif wakeword_ratio < 0.5:
        # More non-wakewords than wakewords
        # Use all available wakewords if possible
        n_wake_target = min(n_wake, int(n_non * wakeword_ratio / (1 - wakeword_ratio)))
        n_non_target = int(n_wake_target * (1 - wakeword_ratio) / wakeword_ratio)
    else:
        # More wakewords than non-wakewords
        # Use all available non-wakewords if possible
        n_non_target = min(n_non, int(n_wake * (1 - wakeword_ratio) / wakeword_ratio))
        n_wake_target = int(n_non_target * wakeword_ratio / (1 - wakeword_ratio))

    # Ensure we don't exceed available samples
    n_wake_target = min(n_wake_target, n_wake)
    n_non_target = min(n_non_target, n_non)

    print(f"[INFO] Target - WakeWords: {n_wake_target}, Non-WakeWords: {n_non_target}")

    # Random selection
    np.random.shuffle(idx_wake)
    np.random.shuffle(idx_non)

    idx_wake_bal = idx_wake[:n_wake_target]
    idx_non_bal = idx_non[:n_non_target]

    # Combine and shuffle
    idx_balanced = np.concatenate([idx_wake_bal, idx_non_bal])
    np.random.shuffle(idx_balanced)

    # Balanced data
    X = X[idx_balanced]
    y = y[idx_balanced]

    final_wake = sum(y == 1)
    final_non = sum(y == 0)
    final_ratio = final_wake / len(y) if len(y) > 0 else 0

    print(f"[INFO] Balanced dataset size: {len(y)}")
    print(f"[INFO] Final distribution - wake={final_wake}, non={final_non}, ratio={final_ratio:.2f}")

    return X, y
