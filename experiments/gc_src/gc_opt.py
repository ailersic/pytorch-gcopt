import torch 
import torch.nn as nn
import numpy as np
import copy
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# GCOptimizer Class:
class GCOptimizer(Optimizer):
    
    # Init Method: 
    def __init__(self,
                base_optimizer: Optimizer,
                cont_var0: float = float('nan'),
                cont_lr: float = 1e-2,
                cont_reg: float = 0.0
                ):
        super(GCOptimizer, self).__init__(base_optimizer.param_groups, base_optimizer.defaults) 

        self.base_optimizer = base_optimizer
        base_optimizer_class = base_optimizer.__class__

        logcontvar0 = np.log(cont_var0)
        if np.isnan(logcontvar0):
            self.logcontvar = torch.tensor(logcontvar0)
        else:
            self.logcontvar0 = logcontvar0
            self.logcontvar = torch.nn.Parameter(torch.tensor(logcontvar0))
            self.contvar_optimizer = base_optimizer_class((self.logcontvar,), lr=cont_lr)
        
        self.cont_lr = cont_lr
        self.cont_reg = cont_reg
        self.warmup = False

        self.rand_samp = self._get_param_groups()
        self.ref_param_groups = self._get_param_groups()
        self.perturbed_param_groups = self._get_param_groups()

        self.n_params = sum(sum(p.numel() for p in group['params']) for group in self.param_groups)

    def _get_param_groups(self):
        param_groups = [group['params'] for group in self.param_groups]
        return copy.deepcopy(param_groups)

    def _set_param_groups(self, new_param_groups):
        for n_g, group in enumerate(self.param_groups):
            for n_p, param in enumerate(group['params']):
                param.data = new_param_groups[n_g][n_p].data

    def _perturb_params(self):
        if not torch.isnan(self.logcontvar):
            # save current parameters
            self.ref_param_groups = self._get_param_groups()

            # generate perturbed parameters
            contvar = torch.exp(torch.clamp(self.logcontvar, max=self.logcontvar0 + 2.0))
            stdev = torch.sqrt(contvar)

            for n_g, group in enumerate(self.ref_param_groups):
                for n_p, param in enumerate(group):
                    self.rand_samp[n_g][n_p] = torch.randn(param.shape, device=param.device, dtype=param.dtype)
                    delta_param = stdev*self.rand_samp[n_g][n_p]
                    self.perturbed_param_groups[n_g][n_p] = param - delta_param

            # set perturbed parameters
            self._set_param_groups(self.perturbed_param_groups)

    def _unperturb_params(self):
        # reload reference parameters
        self._set_param_groups(self.ref_param_groups)

    def _set_contvar_grad(self, contvar_grad: torch.Tensor):
        if type(contvar_grad) is not torch.Tensor:
            contvar_grad = torch.tensor(contvar_grad)
        if contvar_grad.dtype is not torch.float64:
            contvar_grad = contvar_grad.to(torch.float64)
        self.logcontvar.grad = contvar_grad

    def _calc_contvar_grad(self):
        # pathwise gradient estimator
        stdev = torch.sqrt(torch.exp(self.logcontvar))
        for n_g, group in enumerate(self.param_groups):
            for n_p, param in enumerate(group['params']):
                if n_g == 0 and n_p == 0:
                    logcontvar_grad = -(self.rand_samp[n_g][n_p]*param.grad).sum()*stdev/2.0
                else:
                    logcontvar_grad += -(self.rand_samp[n_g][n_p]*param.grad).sum()*stdev/2.0
    
        # add regularization
        logcontvar_grad += self.cont_reg*self.n_params*torch.exp(self.logcontvar - self.logcontvar0)

        return logcontvar_grad.item()

    # Zero Grad Method
    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
        if not torch.isnan(self.logcontvar):
            self.logcontvar.grad = None

    # Step Method 
    def step(self, closure: callable):
        if closure is None:
            raise Exception("Closure is required for GCOptimizer")
        closure = torch.enable_grad()(closure)

        if not torch.isnan(self.logcontvar):
            with torch.no_grad():
                self._perturb_params()

        loss = closure()
        
        if not torch.isnan(self.logcontvar):
            if not self.warmup:
                contvar_grad = self._calc_contvar_grad()
                self._set_contvar_grad(contvar_grad)
            with torch.no_grad():
                self._unperturb_params()

        self.base_optimizer.step()
        if not torch.isnan(self.logcontvar):
            self.contvar_optimizer.step()

        return loss

# Warmup scheduler
class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs = 0):
        self.warmup_epochs = warmup_epochs
        super(WarmupScheduler, self).__init__(optimizer)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        if epoch < self.warmup_epochs:
            self.optimizer.warmup = True
        else:
            self.optimizer.warmup = False
