import torch

from loss import *
import numpy as np


def train_ct(args, epoch, G, P, g_opt, p_opt, dataloader, device, writer):
    # initialization of navgator projection loss and generator loss
    n_loss = 0
    g_loss = 0
    for i, X in enumerate(dataloader):    
        N = X.size(0)
        X = X.view(-1, args.x_dim).to(device)
        rho = args.rho
        # generate samples B x d
        z = torch.randn(N, args.z_dim).to(device)
        xpred = G(z)
        
        X_ = X.repeat(N,1,1).transpose(0,1)
        xpred_ = xpred.repeat(N,1,1)
        
        diff = (X[:,None]-xpred).pow(2) #pairwise mse for navigator network: B x B x h 
        cost = diff.sum(-1) #pairwise cost: B x B
        tmp = P(diff).squeeze() # navigator distance: B x B
        m_backward = torch.nn.functional.softmax(tmp, dim=0) # backward map
        m_forward = torch.nn.functional.softmax(tmp, dim=1) # forward map
        
        p_opt.zero_grad()
        g_opt.zero_grad()
        gloss = (cost * m_forward).sum(1).mean() # forward transport
        nloss = (cost * m_backward).sum(0).mean() # backward transport
        loss = rho * gloss + (1-rho) * nloss
        loss.backward()
        g_opt.step()
        p_opt.step()
        g_loss += gloss.item()
        n_loss += nloss.item()
        
        n_iter = epoch * len(dataloader) + i
        writer.add_scalar('Loss/Generator', gloss.item(), n_iter)
        writer.add_scalar('Loss/Sorter', nloss.item(), n_iter)


    g_loss /= i+1
    n_loss /= i+1
    
    if epoch % 100 == 0:
        print("{} {} Epoch {}: \t nloss {}  \t gloss {} \t ".format(args.method, args.dataset, epoch, nloss.item(), gloss.item()))
    return nloss.item(), gloss.item()

def train_ct_withD(args, epoch, G, P, D, g_opt, p_opt, d_opt, dataloader, device, writer):
    n_loss = 0
    g_loss = 0
    for i, X in enumerate(dataloader):    
        N = X.size(0)
        X = X.view(-1, args.x_dim).to(device)
        rho = args.rho

        z = torch.randn(N, args.z_dim).to(device)

        # ----------------------------------------
        # Update Generator and Navigator network: minimize ct loss
        xpred = G(z)        
        xpred_feat = D(xpred) # feature of generations: B x d
        x_feat = D(X) # feature of data: B x d
        
        cost = torch.norm(x_feat[:,None]-xpred_feat, dim=-1).pow(2) #pairwise cost: B x B
        diff = (x_feat[:,None]-xpred_feat).pow(2)  #pairwise mse for navigator network: B x B x d
        tmp = P(diff).squeeze()  # navigator distance: B x B
        weight_x = torch.nn.functional.softmax(tmp, dim=0) # backward map
        weight_xpred = torch.nn.functional.softmax(tmp, dim=1) # forward map
        
        g_opt.zero_grad()
        p_opt.zero_grad()
        gloss = (cost * weight_xpred).sum(1).mean() # forward transport
        nloss = (cost * weight_x).sum(0).mean() # backward transport
        loss = rho * gloss + (1-rho) * nloss
        loss.backward()
        g_opt.step()
        p_opt.step()
        
        n_loss += nloss.item()
        g_loss += gloss.item()
        
        
        # ----------------------------------------
        # Update Critic network D: maximize ct loss
        xpred = G(z)
        xpred_feat = D(xpred)
        x_feat = D(X)
        
        cost = torch.norm(x_feat[:,None]-xpred_feat, dim=-1).pow(2)

        diff = (x_feat[:,None]-xpred_feat).pow(2)
        tmp = P(diff).squeeze()
        weight_x = torch.nn.functional.softmax(tmp, dim=0) # backward map
        weight_xpred = torch.nn.functional.softmax(tmp, dim=1) # forward map

        d_opt.zero_grad()
        dloss = -((1-rho)*(cost * weight_x).sum(0).mean() + rho*(cost * weight_xpred).sum(1).mean())
        dloss.backward()
        d_opt.step()
        n_iter = epoch * len(dataloader) + i
        writer.add_scalar('Loss/Generator', gloss.item(), n_iter)
        writer.add_scalar('Loss/Sorter', nloss.item(), n_iter)
        

    g_loss /= i+1
    n_loss /= i+1
    
    if epoch % 100 == 0:
        print("{} {} Epoch {}: \t nloss {}  \t gloss {} \t ".format(args.method, args.dataset, epoch, nloss.item(), gloss.item()))
    return nloss.item(), gloss.item()


def train_GAN(args, epoch, G, D, g_opt, d_opt, d_criterion, dataloader, device, writer):
    d_loss = 0
    g_loss = 0
    for i, X in enumerate(dataloader):
        N = X.size(0)
        X = X.view(-1, args.x_dim).to(device)
        
        z = torch.randn(N, args.z_dim).to(device)
        xpred = G(z).view(-1, args.x_dim)
        xpred_1d =  D(xpred)
        x_1d = D(X)

        g_opt.zero_grad()
        gloss = d_criterion(xpred_1d, torch.ones_like(xpred_1d))
        gloss.backward()
        g_opt.step()
        g_loss += gloss.item()
        
        
        z = torch.randn(N, args.z_dim).to(device)
        xpred = G(z).view(-1, args.x_dim)
        xpred_1d =  D(xpred)
        x_1d = D(X)
        dloss_fake = d_criterion(xpred_1d, torch.zeros_like(xpred_1d))
        dloss_true = d_criterion(x_1d, torch.ones_like(x_1d))
        dloss = dloss_fake + dloss_true
        if epoch <= 5000:
            d_opt.zero_grad()
            dloss.backward()
            d_opt.step()

        d_loss += dloss.item()
        n_iter = epoch * len(dataloader) + i
        writer.add_scalar('Loss/Generator', gloss.item(), n_iter)
        writer.add_scalar('Loss/Sorter', dloss.item(), n_iter)


    g_loss /= i+1
    d_loss /= i+1
    
    if epoch % 100 == 0:
        print("{} {} Epoch {}: \t dloss {}  \t gloss {}".format(args.method, args.dataset, epoch, dloss.item(), gloss.item()))
    return dloss.item(), gloss.item()


def train_SWD(args, epoch, G, g_opt, dataloader, device, writer):
    g_loss = 0
    for i, X in enumerate(dataloader):
        g_opt.zero_grad()
        N = X.size(0)
        X = X.view(-1, args.x_dim).to(device)
        z = torch.randn(N, args.z_dim).to(device)
        theta = torch.randn((args.x_dim, args.n_projections),
                            requires_grad=False,
                            device=device)
        theta = theta/torch.norm(theta, dim=0)[None, :]
        xpred = G(z).view(-1, args.x_dim)
        xpred_1d = xpred@theta
        x_1d = X@theta

        gloss = wasserstein1d(xpred_1d, x_1d)
        gloss.backward()
        g_opt.step()
        g_loss += gloss.item()

        n_iter = epoch * len(dataloader) + i
        writer.add_scalar('Loss/Generator', gloss.item(), n_iter)


    g_loss /= i+1
    
    if epoch % 100 == 0:
        print("{} {} Epoch {}: \t gloss {}".format(args.method, args.dataset, epoch, gloss.item()))
    return gloss.item()


def train_WGANGP(args, epoch, G, D, g_opt, d_opt, dataloader, device, writer):
    d_loss = 0
    g_loss = 0
    one = torch.FloatTensor([1]).to(device)
    mone = one * -1
    
    
    for i, X in enumerate(dataloader):
        N = X.size(0)
        X = X.view(-1, args.x_dim).to(device)
        
        z = torch.randn(N, args.z_dim).to(device)
        xpred = G(z).view(-1, args.x_dim)
        
        
        xpred_1d = D(xpred)
        x_1d = D(X)
        
        
        d_opt.zero_grad()
        D_real = x_1d.mean()
        
        D_fake = xpred_1d.mean()
        gradient_penalty = calc_gradient_penalty(D, X, xpred)

        dloss = D_fake - D_real + gradient_penalty
        dloss.backward()
        d_opt.step()

        
        z = torch.randn(N, args.z_dim).to(device)
        xpred = G(z).view(-1, args.x_dim)
        xpred_1d = D(xpred)
        x_1d = D(X)
        
        g_opt.zero_grad()
        g_loss = -xpred_1d.mean()
        g_loss.backward()
        gloss = -g_loss
        if i % 5 == 0:
            g_opt.step()
        
        n_iter = epoch * len(dataloader) + i
        writer.add_scalar('Loss/Generator', gloss.item(), n_iter)
        writer.add_scalar('Loss/Sorter', dloss.item(), n_iter)


    g_loss /= i+1
    d_loss /= i+1
    
    if epoch % 100 == 0:
        print("{} {} Epoch {}: \t dloss {}  \t gloss {}".format(args.method, args.dataset, epoch, dloss.item(), gloss.item()))
    return dloss.item(), gloss.item()

