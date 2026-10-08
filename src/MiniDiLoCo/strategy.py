import torch
from torch.optim import AdamW, SGD
from transformers import get_cosine_schedule_with_warmup
from abc import ABC, abstractmethod
from typing import override, Callable, Type
from torch.optim import Optimizer

class Strategy(ABC):
    """ Base class for training strategy. """

    def _init_node(self, model, rank, world_size):
        self.model = model
        self.rank = rank
        self.world_size = world_size

    @abstractmethod
    def step(self, batch):
        pass

class Diloco(Strategy):
    """     DiLoCo PyTorch strategy.     """
    """ https://arxiv.org/pdf/2311.08105 """

    def __init__(self,
                 inner_lr: float = 4e-4,
                 outer_lr: float = 0.7,
                 warmup_steps: int = 50, #1000
                 weight_decay: float = 0.1,
                 H: int = 500,
                 betas: tuple = (0.9, 0.95),
                 momentum: float = 0.9,
                 eps: float = 10e-1,
                 inner_optimizer_cls: Type[Optimizer] = AdamW,
                 inner_optimizer_kwargs: dict = None,
                 outer_optimizer_cls: Type[Optimizer] = SGD,
                 outer_optimizer_kwargs: dict = None,
                 loss_fn: Callable = None,
                 ):
        """ The parameters come from Table 5 of the paper. """

        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.H = H
        self.betas = betas
        self.momentum = momentum
        self.eps = eps

        self.loss_fn = loss_fn
        self.inner_optimizer_cls    = inner_optimizer_cls    or torch.optim.AdamW
        self.inner_optimizer_kwargs = inner_optimizer_kwargs or {"lr": inner_lr}
        self.outer_optimizer_cls    = outer_optimizer_cls    or torch.optim.SGD
        self.outer_optimizer_kwargs = outer_optimizer_kwargs or {"lr": 0.7, "nesterov": True, "momentum": 0.9}

    @override
    def _init_node(self, model, rank, world_size, total_steps=100):
        """ Initialize the optimizers and scheduler of DiLoCo """

        super()._init_node(model, rank, world_size)

        local_params = list(model.parameters())
        global_params = [p.clone().detach().requires_grad_(True) for p in model.parameters()]

        self.inner_opt = self.inner_optimizer_cls(local_params, **self.inner_optimizer_kwargs)
        self.outer_opt = self.outer_optimizer_cls(global_params, **self.outer_optimizer_kwargs)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.inner_opt,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps * self.H
        )

        print(f"Rank {self.rank}: DiLoCo H={self.H}")

    def step(self, batch):
        """ Inner loop of the DiLoCo strategy """

        self.inner_opt.zero_grad()

        logits = self.model(**batch)
        loss = self.loss_fn(logits, batch["labels"])

        loss.backward()
        self.inner_opt.step()
        self.scheduler.step()
        return loss.item()