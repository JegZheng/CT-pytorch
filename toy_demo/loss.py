import torch
import torch.nn as nn
import numpy as np
import torch.autograd as autograd

# loss for Sliced Wasserstein Generator
def wasserstein1d(x, y):
    n = x.size(0)
    x1, _ = torch.sort(x, dim=0)
    y1, _ = torch.sort(y, dim=0)
    z = (x1-y1)
    return (torch.norm(z,dim=0)**2).mean() 


# gradient penalty for WGAN-GP
def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(real_data.device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    
    interpolates = interpolates.to(real_data.device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(real_data.device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * .1
    return gradient_penalty