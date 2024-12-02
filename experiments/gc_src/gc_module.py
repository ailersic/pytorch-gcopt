import torch
import copy
import pytorch_lightning as pl
import numpy as np

class ContNet(pl.LightningModule):
    def __init__(self,
                logcontvar0: float = float('nan'),
                loggradvar0: float = float('nan'),
                cont_lr: float = 1e-2,
                cont_reg: float = 0.0,
                warmup_epochs: int = 0
                ):
        super().__init__()

        if np.isnan(logcontvar0):
            self.logcontvar = torch.tensor(logcontvar0)
        else:
            self.logcontvar0 = logcontvar0
            self.logcontvar = torch.nn.Parameter(torch.tensor(logcontvar0))
        
        if np.isnan(loggradvar0):
            self.loggradvar = torch.tensor(loggradvar0)
        else:
            self.loggradvar0 = loggradvar0
            self.loggradvar = torch.nn.Parameter(torch.tensor(loggradvar0))
        
        self.cont_lr = cont_lr
        self.warmup_epochs = warmup_epochs
        self.cont_reg = cont_reg

        self.rand_samp = None
        self.ref_params = None
        self.perturbed_params = None

        self.automatic_optimization = False
    
    def reperturb_params(self):
        if not torch.isnan(self.logcontvar):
            # reload previously generated perturbed params
            self.load_state_dict(self.perturbed_params)

    def _generate_perturbed_params(self, contvar: torch.Tensor):
        rand_samp = {}
        perturb_params = {}

        for pname in self.ref_params:
            if str(pname).startswith('logcontvar'):
                perturb_params[pname] = self.ref_params[pname]
            else:
                rand_samp[pname] = torch.randn(self.ref_params[pname].shape, device=self.device)
                delta_param = torch.sqrt(contvar)*rand_samp[pname]
                perturb_params[pname] = self.ref_params[pname].to(self.device) - delta_param.to(self.device)
        
        return rand_samp, perturb_params

    def perturb_params(self):
        if not torch.isnan(self.logcontvar):
            self.ref_params = copy.deepcopy(self.state_dict())
            contvar = torch.exp(torch.clamp(self.logcontvar, max=self.logcontvar0 + 2.0))

            self.log('contvar', contvar, prog_bar=True)

            # generate and load perturbed parameters
            self.rand_samp, self.perturbed_params = self._generate_perturbed_params(contvar)
            self.load_state_dict(self.perturbed_params)
    
    def perturb_grad(self):
        if not torch.isnan(self.loggradvar):
            gradvar = torch.exp(self.loggradvar)

            self.log('gradvar', gradvar, prog_bar=True)

            # generate and load perturbed parameters
            for pname, p in self.named_parameters():
                if str(pname).startswith('logcontvar'):
                    continue
                if p.grad is not None:
                    delta_grad = torch.sqrt(gradvar)*torch.randn(p.grad.shape, device=self.device)
                    p.grad = p.grad.to(self.device) + delta_grad.to(self.device)
    
    def unperturb_params(self):
        if not torch.isnan(self.logcontvar):
            # reload reference parameters
            self.load_state_dict(self.ref_params)

    def set_contvar_grad(self, contvar_grad: torch.Tensor):
        if not torch.isnan(self.logcontvar):
            if type(contvar_grad) is not torch.Tensor:
                contvar_grad = torch.tensor(contvar_grad, device=self.device)
            if contvar_grad.dtype is not torch.float64:
                contvar_grad = contvar_grad.to(torch.float64)
            self.logcontvar.grad = contvar_grad.to(self.device)

    def calc_contvar_grad(self, loss: torch.Tensor = None, reinforce: bool = False):
        if (self.current_epoch >= self.warmup_epochs) and (not torch.isnan(self.logcontvar)):
            n_params = sum(p.numel() for p in self.parameters()) - 1

            # compute gradient of logcontvar
            if reinforce:
                if loss is None:
                    raise ValueError('loss must be provided for reinforce gradient estimator')
                logcontvar_grad = -n_params/2.0
                for pname in rand_samp:
                    logcontvar_grad += self.rand_samp[pname].pow(2.0).sum()/2.0
                logcontvar_grad *= loss.detach()
            else:
                stdev = torch.sqrt(torch.exp(self.logcontvar))
                logcontvar_grad = torch.tensor(0.0, device=self.device)
                for pname, param in self.named_parameters():
                    if pname == 'logcontvar' or param.grad is None:
                        continue
                    logcontvar_grad += -(self.rand_samp[pname]*param.grad).sum()*stdev/2.0
        
            # add regularization
            logcontvar_grad += self.cont_reg*n_params*torch.exp(self.logcontvar - self.logcontvar0)

            self.log('logcontvar_grad', logcontvar_grad)
            return logcontvar_grad.item()
            
        else:
            return 0.0