import torch
from torch.optim import Optimizer


class CLMR(Optimizer):
    """
    Cyclic Learning/Momentum Rate optimizer.
    Arxiv: https://arxiv.org/abs/2302.02289

    SGD avec LR cyclique et momentum rate cyclique inversé simultanément.
    Quand le LR monte, le momentum descend (et vice versa).
    Yields >2% improvement in Dice vs SGD/CLR on medical segmentation.
    """

    def __init__(self, params, base_lr=1e-4, max_lr=1e-2,
                 base_momentum=0.85, max_momentum=0.95,
                 cycle_size=20, weight_decay=0.0):
        if base_lr < 0:
            raise ValueError(f"base_lr invalide: {base_lr}")
        if not 0.0 <= base_momentum <= 1.0:
            raise ValueError(f"base_momentum invalide: {base_momentum}")
        defaults = dict(
            lr=base_lr,
            base_lr=base_lr, max_lr=max_lr,
            base_momentum=base_momentum, max_momentum=max_momentum,
            momentum=max_momentum,
            cycle_size=cycle_size,
            weight_decay=weight_decay,
            step_count=0,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            step = group['step_count']
            cycle = step // group['cycle_size']
            # position dans le cycle : 0 → 1 → 0 (triangulaire)
            x = abs(step / group['cycle_size'] - cycle - 0.5) * 2

            # LR monte → momentum descend (inversé)
            group['lr'] = group['base_lr'] + (group['max_lr'] - group['base_lr']) * x
            group['momentum'] = group['max_momentum'] - (group['max_momentum'] - group['base_momentum']) * x
            group['step_count'] += 1

            lr = group['lr']
            momentum = group['momentum']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if wd != 0.0:
                    d_p = d_p.add(p, alpha=wd)

                state = self.state[p]
                if 'buf' not in state:
                    state['buf'] = torch.zeros_like(p)
                buf = state['buf']
                buf.mul_(momentum).add_(d_p)
                p.add_(buf, alpha=-lr)

        return loss