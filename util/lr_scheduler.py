# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import bisect

import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler
from torch.optim.lr_scheduler import OneCycleLR

def build_scheduler(train_cfg, optimizer, min_lr=1e-5, warmup_lr=1e-5):
    num_steps = int(train_cfg.num_epoch * train_cfg.epoch_iters)
    warmup_steps = int(train_cfg.LR_SCHEDULER.warmup_epochs * train_cfg.epoch_iters)
    decay_steps = int(train_cfg.LR_SCHEDULER.decay_epochs * train_cfg.epoch_iters)
    n_epochs = train_cfg.end_epoch - train_cfg.start_epoch

    lr_scheduler = None
    if train_cfg.LR_SCHEDULER.name == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=(num_steps - warmup_steps) if train_cfg.LR_SCHEDULER.warmup else num_steps,
            lr_min=min_lr,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
            warmup_prefix=train_cfg.LR_SCHEDULER.warmup,
        )
    elif train_cfg.LR_SCHEDULER.name == 'onecycle':
        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=[i['lr'] for i in optimizer.param_groups],
            total_steps=train_cfg.end_epoch,
            div_factor=25,
            pct_start=0.1,
            anneal_strategy='cos',
            final_div_factor=1e4
        )
    elif train_cfg.LR_SCHEDULER.name == 'linear':
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            warmup_lr_init=train_cfg.warmup_lr,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif train_cfg.LR_SCHEDULER.NAME == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=train_cfg.LR_SCHEDULER.DECAY_RATE,
            warmup_lr_init=train_cfg.warmup_lr,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )

    return lr_scheduler


class LinearLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None
