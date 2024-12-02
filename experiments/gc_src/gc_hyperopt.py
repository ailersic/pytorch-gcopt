import torch 
import torch.nn as nn
import numpy as np
import copy
from torch.optim import Optimizer

torch.set_grad_enabled(True)

class GCHyperOptimizer(Optimizer):
    def __init__(self,
                base_opt: Optimizer,
                contvar0: torch.Tensor = torch.tensor([1e-4]),
                contvar_lr: float = 1e-2,
                ):
        defaults = dict(base_opt=base_opt, contvar0=contvar0, contvar_lr=contvar_lr)
        super(GCHyperOptimizer, self).__init__(base_opt.param_groups, defaults)

        self.base_opt = base_opt

        opt_class = base_opt.__class__
        self.logcontvar = nn.Parameter(torch.log(contvar0), requires_grad=True)
        self.contvar_opt = opt_class([self.logcontvar], lr=contvar_lr)

    def step(self, closure=None):
        rand_p = []
        stdev = torch.sqrt(torch.exp(self.logcontvar))

        # Perturb the parameters
        for j, p in enumerate(self.param_groups[0]['params']):
            rand_p.append(torch.randn_like(p))
            with torch.no_grad():
                p.add_(-stdev * rand_p[j])

        # Compute the gradient
        self.logcontvar.grad = None  # Clear the gradient before computing it
        loss = closure()

        # Unperturb the parameters
        for j, p in enumerate(self.param_groups[0]['params']):
            with torch.no_grad():
                p.add_(stdev * rand_p[j])

        # Calculate logcontvar gradient
        logcontvar_grad = torch.zeros_like(self.logcontvar)
        for j, p in enumerate(self.param_groups[0]['params']):
            logcontvar_grad += torch.tensordot(p.grad, -0.5*stdev*rand_p[j], dims=len(p.shape))
        self.logcontvar.grad = logcontvar_grad

        #print(self.logcontvar.grad, self.logcontvar.data)

        # Opt step
        self.base_opt.step()
        self.contvar_opt.step()

        return loss
