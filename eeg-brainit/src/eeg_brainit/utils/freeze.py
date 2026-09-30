from __future__ import annotations

import torch.nn as nn


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def freeze(module: nn.Module) -> None:
    module.eval()
    set_requires_grad(module, False)


def unfreeze(module: nn.Module) -> None:
    set_requires_grad(module, True)
    module.train()


def count_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
