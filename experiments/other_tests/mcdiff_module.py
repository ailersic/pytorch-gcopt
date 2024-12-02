import torch
import pytorch_lightning as pl

class MCDiffNet(pl.LightningModule):
    def __init__(self, sigma, n_mc):
        super().__init__()

        self.sigma = sigma
        self.n_mc = n_mc

        self.automatic_optimization = False

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        self.log('param_norm', sum(p.pow(2.0).sum() for p in self.net.parameters()))

        mean_loss = 0
        ref_params = self.net.state_dict()
        
        opt = self.optimizers()
        opt.zero_grad()
        for param in self.parameters():
            param.grad = torch.zeros_like(param.data)

        for i in range(self.n_mc):
            delta_param = {}
            perturb_params = {}
            for pname in ref_params:
                delta_param[pname] = self.sigma*torch.randn(ref_params[pname].shape).cuda()
                perturb_params[pname] = ref_params[pname] - delta_param[pname]

            self.net.load_state_dict(perturb_params)
            ypred = self.forward(x)

            loss = self.lossfunc(ypred, y)/self.n_mc
            mean_loss += loss.detach()
            
            # not doing backprop
            #self.manual_backward(loss)

            # doing monte carlo gradient instead
            for pname, param in self.net.named_parameters():
                param.grad += -(delta_param[pname]/(self.sigma**2))*loss
        
        opt.step()
        self.log('train_loss', mean_loss)
        
        self.net.load_state_dict(ref_params)
        ypred = self.forward(x)
        real_loss = self.lossfunc(ypred, y)

        self.log('real_train_loss', real_loss)