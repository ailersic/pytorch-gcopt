# PyTorch-GCOpt
This is the PyTorch implementation of the Gaussian continuation optimizer. It accompanies [this paper](https://www.aimsciences.org/article/doi/10.3934/fods.2024053) by Andrew F. Ilersich and Prasanth B. Nair. The code used to produce the results in the paper is in the `experiments` directory.

If GCOptimizer is used in your work, please cite:
```
@article{Ilersich_Nair_2024_Gauss_Cont,
    title = {Deep learning with Gaussian continuation},
    journal = {Foundations of Data Science},
    pages = {},
    year = {2024},
    issn = {},
    doi = {10.3934/fods.2024053},
    url = {https://www.aimsciences.org/article/id/6761494fd0b64a5d1d351a65},
    author = {Andrew F. Ilersich and Prasanth B. Nair},
    keywords = {Non-convex optimization, deep learning, gradient descent, continuation algorithms}
}
```

## Usage
Download the `gc_opt.py` file and place it in your working directory. Then import the optimizer:
```
from gc_opt import *
```
The next step is to define a base optimizer of your choice, which runs under the hood:
```
base_optimizer = Adam(model.parameters(), lr=1e-4, etc.)
```
Then finally, define your GCOptimizer using the base optimizer:
```
optimizer = GCOptimizer(base_optimizer, cont_var0 = 1e-4, cont_lr = 1e-2, cont_reg = 1e-4)
```
where `cont_var0` is the initial value of the continuation parameter, `cont_lr` is the learning rate on it, and `cont_reg` is the regularization parameter. Setting `cont_var0` to `nan` disables continuation, so your GCOptimizer functions exactly like the base optimizer. Optionally, you may also define a WarmupScheduler for your GCOptimizer:
```
scheduler = WarmupScheduler(optimizer, warmup_epochs = 5)
```
where `warmup_epochs` is how many epochs for which the continuation parameter is held constant.

Unlike SGD, Adam, etc., but like L-BFGS, the `.step()` function for the GCOptimizer takes a callable `closure`, which calculates and returns the loss, as an argument. An example usage is as follows:
```
for epoch in range(n_epochs):
    for i, data in enumerate(trainloader):
        inputs, labels = data

        def closure():
            optimizer.zero_grad()
            outputs = mynet(inputs)
            loss = lossfunc(outputs, labels)
            loss.backward()

            return loss

        loss = optimizer.step(closure)
```
