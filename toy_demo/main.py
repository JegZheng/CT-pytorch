import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import os
from data_loader import load_data
import datetime
from torch.utils.tensorboard import SummaryWriter
from training import *
import seaborn as sns

parser = argparse.ArgumentParser(description='Conditional transport experiment on toydata')
# dataset options
parser.add_argument('--batchsize', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 200)')
parser.add_argument('--dataset', type=str, default="swiss_roll", metavar='D',
                    help='Dataset: swiss_roll|half_moons|circle|s_curve|2gaussians|8gaussians|25gaussians')
parser.add_argument('--toysize', type=int, default=2000, metavar='N',
                    help='toy dataset size for training (default: 2000)')
# training options
parser.add_argument('--method', type=str, default="ACT", metavar='D',
                    help='CT|CT_withD|GAN|SWD|MSWD')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10000)')
parser.add_argument('--z_dim', type=int, default=50, metavar='N',
                    help='dimensionality of z (default: 50)')
parser.add_argument('--x_dim', type=int, default=2, metavar='N',
                    help='dimensionality of x (default: 2)')
parser.add_argument('--p_dim', type=int, default=1, metavar='N',
                    help='dimensionality of projected x (default: 1)')
parser.add_argument('--d_dim', type=int, default=10, metavar='N',
                    help='dimensionality of feature x_feat (default: 20)')
parser.add_argument('--learning-rate', type=float, default=2e-4,
                    help='learning rate for Adam')
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument('--device', type=str, default="0", metavar='D',
                    help='which device for training: 0, 1, 2, 3 (GPU) or cpu')
parser.add_argument('--run_all', action='store_true', default=False, help='activate to run all methods on all datasets')

# CT options
parser.add_argument('--rho', type=float, default=0.5,
                    help='balance coefficient for forward-backward (default: 0.5)')

# SWD options
parser.add_argument('--n_projections', type=int, default=1000, metavar='N',
                    help='number of projections for input x (default: 1000)')

# saving options
parser.add_argument('--remark', type=str, default="experiment1", metavar='R',
                    help='leave some remark for this experiment')
parser.add_argument('--save_fig', action='store_true', default=False, help='activate to save sampled and reconstructed figures')

args = parser.parse_args()
device = 'cuda:' + args.device if torch.cuda.is_available() else 'cpu' 

# Generator architecture
class Generator(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Generator, self).__init__()
        self.model = nn.Sequential(nn.Linear(D_in, H),
                                   nn.BatchNorm1d(H),
                                   nn.LeakyReLU(),
                                   nn.Linear(H, H//2),
                                   nn.BatchNorm1d(H//2),
                                   nn.LeakyReLU(),
                                   torch.nn.Linear(H//2, D_out)
                                  )

    def forward(self, x):
        mu = self.model(x)
        return mu


# Navigator/Discriminator/Feature encoder for CT, GAN and WGAN
class Projector(torch.nn.Module):
    def __init__(self, D_in, H, D_out=1):
        super(Projector, self).__init__()
        self.model = nn.Sequential(nn.Linear(D_in, H),
                                   nn.LeakyReLU(),
                                   nn.Linear(H, H//2),
                                   nn.LeakyReLU(),
                                   nn.Linear(H//2, D_out)
                                  )

    def forward(self, x):
        logit = self.model(x)
        return logit


def main():
    print(args)
    # saving path
    name = args.method + '_' + args.remark + '_' + str(args.dataset) + '_'+ str(datetime.datetime.now()).replace(' ', '_')
    model_path = os.path.join('models', name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    img_path = os.path.join('imgs', args.method + '_' + args.remark)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    
    X_data = load_data(name=args.dataset, n_samples=args.toysize)
    dataloader = data_utils.DataLoader(X_data, shuffle=True,
                                   batch_size=args.batchsize)
    # Tensorboard: optional for visualization
    writer = SummaryWriter(log_dir= os.path.join('runs', 'toy',  name))

    G = Generator(D_in=args.z_dim, H=100, D_out=args.x_dim).to(device)
    g_opt = optim.Adam(G.parameters(), lr=args.learning_rate)

    if args.method =='CT_withD':
        D = Projector(D_in=args.x_dim, H=100, D_out=args.d_dim).to(device)
        d_opt = optim.Adam(D.parameters(), lr=args.learning_rate)
        P = Projector(D_in=args.d_dim, H=100, D_out=args.p_dim).to(device)
    else:
        P = Projector(D_in=args.x_dim, H=100, D_out=args.p_dim).to(device)
    p_opt = optim.Adam(P.parameters(), lr=args.learning_rate/5)
    z_fix = torch.randn(X_data.size(0), args.z_dim, requires_grad=False).to(device)
    
    p_stats = []
    g_stats = []
    swd_stats = []
    # training 
    for epoch in range(args.epochs+1):
        if args.method == 'CT':
            ploss, gloss = train_ct(args, epoch, G, P, g_opt, p_opt, dataloader, device, writer)
            p_stats.append(ploss)
            g_stats.append(gloss)
        elif args.method == 'CT_withD':
            ploss, gloss = train_ct_withD(args, epoch, G, P, D, g_opt, p_opt, d_opt, dataloader, device, writer)
            p_stats.append(ploss)
            g_stats.append(gloss)
        elif args.method == 'GAN':
            d_criterion = nn.BCEWithLogitsLoss()
            ploss, gloss = train_GAN(args, epoch, G, P, g_opt, p_opt, d_criterion, dataloader, device, writer)
            p_stats.append(ploss)
            g_stats.append(gloss)
        elif args.method == 'WGANGP':
            ploss, gloss = train_WGANGP(args, epoch, G, P, g_opt, p_opt, dataloader, device, writer)
            p_stats.append(ploss)
            g_stats.append(gloss)
        elif args.method == 'SWD':
            gloss = train_SWD(args, epoch, G, g_opt, dataloader, device, writer)
            g_stats.append(gloss)
        else:
            raise Exception("Method not found: name must be 'GAN', 'SWD', 'WGANGP', 'CT', 'CT_withD'.")

    # test
        if epoch % 100 == 0:
            with torch.no_grad():
                X = X_data.to(device)
                xpred = G(z_fix)
            # plot true data distribution 
            if epoch == 0:
                fig, (ax) = plt.subplots(1,1,figsize=(6,6))
                sns.kdeplot(X.detach().cpu().numpy()[:,0], X.detach().cpu().numpy()[:,1], ax=ax, cmap="Greens", shade=True, bw=0.1)
                writer.add_figure('data_distribution', fig, epoch)
                fig.savefig(os.path.join(img_path ,'{}_true.pdf'.format(args.dataset)))
                plt.close()
            # plot generated data distribution every 100 epochs 
            if epoch % 100 == 0:
                fig, (ax) = plt.subplots(1,1,figsize=(6,6))
                sns.kdeplot(xpred.detach().cpu().numpy()[:,0], xpred.detach().cpu().numpy()[:,1], ax=ax, cmap="Greens", shade=True, bw=0.1)
                writer.add_figure('test_distribution', fig, epoch)
                fig.savefig(os.path.join(img_path , '{}_fake_{}.pdf'.format(args.dataset,epoch)))
                plt.close()
                
            
    # save training checkpoints and status
    torch.save(G.state_dict(), model_path + name + '.G')
    torch.save(P.state_dict(), model_path + name + '.P')
    torch.save(g_stats, model_path + name + '.gstat')
    if not args.method =='SWD':
        torch.save(p_stats, model_path + name + '.pstat')
    


if __name__ == '__main__':
    if args.run_all:
        methods = ['CT', 'CT_withD', 'GAN', 'WGANGP', 'SWD']
        datasets = ['swiss_roll', 'half_moons', '8gaussians', '25gaussians']
        for method in methods:
            for dataset in datasets:
                args.method = method
                args.dataset = dataset
                main()
    else:
        main()





    