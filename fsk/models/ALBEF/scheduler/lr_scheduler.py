import math
import warnings
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 max_steps: int,
                 warmup_steps: int,
                 warmup_init_lr: float = 0.0,
                 warmup_strategy: str = 'linear',
                 eta_min: float = 0,
                 last_epoch: int = -1,
                 verbose=False):
        """
        Set the learning rate using a cosine annealing schedule with a warmup strategy,
        @param optimizer: Wrapped optimizer
        @param max_steps: Maximum number of steps
        @param warmup_steps: Number of warmup steps
        @param warmup_init_lr: Initial learning rate for warmup
        @param warmup_strategy: Increase the learning rate either linear or constant
        @param eta_min: Minimum learning rate
        @param last_epoch: The index of the last epoch. Default: -1
        @param verbose: If True, prints a message to stdout for each update
        """
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.warmup_init_lr = warmup_init_lr
        self.warmup_strategy = warmup_strategy
        self.eta_min = eta_min

        assert warmup_strategy in ['constant', 'linear'], 'Unknown warmup type for LRScheduler.'
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == self.warmup_steps:
            return self.base_lrs

        elif self.last_epoch == 0:
            return [self.warmup_init_lr for _ in self.optimizer.param_groups]

        elif self.last_epoch < self.warmup_steps:
            if self.warmup_strategy == 'constant':
                return [self.warmup_init_lr for _ in self.base_lrs]
            elif self.warmup_strategy == 'linear':
                return [
                    group["lr"] + (base_lr - self.warmup_init_lr) / (self.warmup_steps - 1)
                    for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
                ]
        elif (self.last_epoch - 1 - self.max_steps) % (2 * self.max_steps) == 0:
            return [group['lr']
                    + (base_lr - self.eta_min)
                    * 0.5
                    * (1 - math.cos(math.pi / (self.max_steps - self.warmup_steps)))
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)))
            / (1 + math.cos(
                math.pi * (self.last_epoch - self.warmup_steps - 1) / (self.max_steps - self.warmup_steps)))
            * (group['lr'] - self.eta_min)
            + self.eta_min
            for group in
            self.optimizer.param_groups]

    def _get_closed_form_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            if self.warmup_strategy == 'constant':
                return [self.warmup_init_lr for _ in self.base_lrs]
            elif self.warmup_strategy == 'linear':
                return [
                    self.warmup_init_lr
                    + self.last_epoch * (base_lr - self.warmup_init_lr) / (self.warmup_steps - 1)
                    for base_lr in self.base_lrs
                ]

        return [self.eta_min
                + 0.5
                * (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) /
                                (self.max_steps - self.warmup_steps)))
                for base_lr in self.base_lrs]
