import torch
import copy
import pytorch_lightning as pl
import numpy as np

class ContNet(pl.LightningModule):
    def __init__(self, logcontvar0: float, n_mc: int, cont_lr: float, cont_reg: float, warmup_epochs: int, lossfunc: callable):
        super().__init__()

        if np.isnan(logcontvar0):
            self.logcontvar = torch.tensor(logcontvar0)
        else:
            self.logcontvar0 = logcontvar0
            self.logcontvar = torch.nn.Parameter(torch.tensor(logcontvar0))
        
        self.n_mc = n_mc
        self.cont_lr = cont_lr
        self.warmup_epochs = warmup_epochs
        self.lossfunc = lossfunc
        self.cont_reg = cont_reg

        self.automatic_optimization = False
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{'params': self.net.parameters()},
                                      {'params': (self.logcontvar,), 'lr': self.cont_lr}],
                                     lr=1e-4)
        return optimizer
    
    def gen_perturb_params(self, ref_params: dict, contvar: float):
        rand_samp = {}
        perturb_params = {}

        for pname in ref_params:
            if str(pname).startswith('logcontvar'):
                perturb_params[pname] = ref_params[pname]
            else:
                #stdev = torch.sqrt(torch.tensor(1/ref_params[pname].shape[-1]))
                rand_samp[pname] = torch.randn(ref_params[pname].shape, device=self.device)
                delta_param = torch.sqrt(contvar)*rand_samp[pname]
                perturb_params[pname] = ref_params[pname] - delta_param

        return rand_samp, perturb_params

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        self.log('param_norm', sum(p.pow(2.0).sum() for p in self.parameters()))
        
        opt = self.optimizers()
        opt.zero_grad()

        if torch.isnan(self.logcontvar):
            ypred = self.forward(x)
            loss = self.lossfunc(ypred, y)
            self.manual_backward(loss)
            
            mean_loss = loss.detach()
            match = torch.eq(y, torch.argmax(ypred, dim=1))
            mean_acc = torch.sum(match.detach())/y.shape[0]

        else:
            mean_loss = 0
            mean_acc = 0

            logcontvar_grad = 0
            ref_params = self.state_dict()
            n_params = sum(p.numel() for p in self.parameters()) - 1

            # compute loss and grad for reference parameters
            '''
            ypred = self.forward(x)
            refloss = self.lossfunc(ypred, y)
            self.manual_backward(refloss)
            refgrad = {}
            for pname, param in self.named_parameters():
                if str(pname).startswith('logcontvar'):
                    refgrad[pname] = torch.zeros_like(param)
                else:
                    refgrad[pname] = param.grad.detach().clone()
            refloss = refloss.detach()

            # reset gradients
            opt.zero_grad()
            '''

            contvar = torch.exp(torch.clamp(self.logcontvar, max=self.logcontvar0 + 2.0))

            self.log('logcontvar', self.logcontvar)
            self.log('contvar', contvar)

            for i in range(self.n_mc):
                # generate and load perturbed parameters
                rand_samp, perturb_params = self.gen_perturb_params(ref_params, contvar)
                self.load_state_dict(perturb_params)

                # compute perturbed loss
                ypred = self.forward(x)
                loss = self.lossfunc(ypred, y)/self.n_mc
                self.manual_backward(loss)

                # compute gradient of logcontvar
                logcontvar_grad_term = -n_params/2.0
                for pname in rand_samp:
                    logcontvar_grad_term += rand_samp[pname].pow(2.0).sum()/2.0
                logcontvar_grad_term *= loss.detach()

                # control variate term
                #for pname in rand_samp:
                #    logcontvar_grad_term -= (n_params*torch.sqrt(contvar)/2.0)*(refgrad[pname]*rand_samp[pname]).sum()
                
                logcontvar_grad += logcontvar_grad_term

                # add to mean over n_mc samples
                mean_loss += loss.detach()
                match = torch.eq(y, torch.argmax(ypred, dim=1))
                acc = torch.sum(match)/y.shape[0]
                mean_acc += acc.detach()/self.n_mc
            
            self.load_state_dict(ref_params)
            if self.current_epoch > self.warmup_epochs:
                self.logcontvar.grad = logcontvar_grad + self.cont_reg*n_params*torch.exp(self.logcontvar - self.logcontvar0)
            self.log('logcontvar_grad', logcontvar_grad)

        opt.step()
        self.log('train_loss', mean_loss)
        self.log('train_acc', mean_acc)
