import torch


class EarlyStopping:
    """Early stops the training if validation metric doesn't improve after a given patience."""

    def __init__(self, patience=5, min_delta=0, mode='max', verbose=True):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change in monitored metric to qualify as improvement.
            mode (str): 'min' for loss, 'max' for accuracy.
            verbose (bool): If True, prints a message for each improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

        if mode == 'min':
            self.monitor_op = lambda current, best: current < best - min_delta
        else:  # mode == 'max'
            self.monitor_op = lambda current, best: current > best + min_delta

    def __call__(self, current_score, epoch):
        """
        Args:
            current_score: The current validation metric value
            epoch: Current epoch number

        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            if self.verbose:
                print(f'EarlyStopping: Initial score set to {current_score:.4f}')
            return False

        if self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f'EarlyStopping: Metric improved to {current_score:.4f}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping: No improvement for {self.counter}/{self.patience} epochs')

            if self.counter >= self.patience:
                if self.verbose:
                    print(f'EarlyStopping: Stopping training. Best score: {self.best_score:.4f} at epoch {self.best_epoch + 1}')
                self.early_stop = True
                return True

        return False

    def get_best_score(self):
        """Returns the best score achieved during training"""
        return self.best_score

    def get_best_epoch(self):
        """Returns the epoch number where best score was achieved"""
        return self.best_epoch
