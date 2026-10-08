from abc import ABC, abstractmethod
from typing import override
import torch.distributed as dist

class Aggregator(ABC):
    def __init__ (self, outer_opt, inner_opt):
        self.outer_opt = outer_opt
        self.inner_opt = inner_opt

    @abstractmethod
    def aggregate(self):
        pass

class AllReduce(Aggregator):
    def __init__(self, outer_param, inner_param):
        super().__init__(outer_param, inner_param)

    @override
    def aggregate(self):
        outer_params = [param for group in self.outer_opt.param_groups for param in group['params']]
        inner_params = [param for group in self.inner_opt.param_groups for param in group['params']]

        for outer_p, inner_p in zip(outer_params, inner_params):
            delta = outer_p.detach() - inner_p.detach()
            dist.all_reduce(tensor=delta, op=dist.ReduceOp.AVG)
            outer_p.grad = delta