import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
import model_resnet
import model

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--disc_iters', type=int, default=1)
parser.add_argument('--gen_iters', type=int, default=5)
parser.add_argument('--epochs', type=int, default=5000)
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--rho', type=float, default=0.5)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--out_dir', type=str, default='out')


parser.add_argument('--model' , type=str, default='resnet')

args = parser.parse_args()

loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data/', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True, num_workers=8)

Z_dim = 128

# discriminator = torch.nn.DataParallel(Discriminator()).cuda() # TODO: try out multi-gpu training
if args.model == 'resnet':
    discriminator = model_resnet.Discriminator(args.dim, loss_type=args.loss).cuda()
    discriminator = torch.nn.DataParallel(discriminator)
    generator = model_resnet.Generator(Z_dim).cuda()
    generator = torch.nn.DataParallel(generator)
else:
    discriminator = model.Discriminator(args.dim, loss_type=args.loss).cuda()
    discriminator = torch.nn.DataParallel(discriminator)
    generator = model.Generator(Z_dim).cuda()
    generator = torch.nn.DataParallel(generator)
navigator = model.Navigator(dim=args.dim).cuda()
navigator = torch.nn.DataParallel(navigator)
# because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to 
# optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(0.0,0.9))
optim_gen  = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0,0.9))
optim_nav  = optim.Adam(navigator.parameters(), lr=args.lr, betas=(0.0,0.9))

# use an exponentially decaying learning rate
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)
scheduler_n = optim.lr_scheduler.ExponentialLR(optim_nav, gamma=0.99)

def train(epoch):
    for batch_idx, (data, target) in enumerate(loader):
        if data.size()[0] != args.batch_size:
            continue
        data, target = (data.cuda()), (target.cuda())

        # update discriminator
        for _ in range(args.disc_iters):
            z = (torch.randn(args.batch_size, Z_dim).cuda())
            optim_disc.zero_grad()
            optim_gen.zero_grad()
            if args.loss == 'hinge':
                disc_loss = nn.ReLU()(1.0 - discriminator(data)).mean() + nn.ReLU()(1.0 + discriminator(generator(z))).mean()
            elif args.loss == 'wasserstein':
                disc_loss = -discriminator(data).mean() + discriminator(generator(z)).mean()
            elif args.loss == 'ct':
                feat = discriminator(generator(z))
                feat_x = discriminator(data)
                mse_n = (feat_x[:,None] - feat).pow(2)
                cost = mse_n.sum(-1)
                d = navigator(mse_n).squeeze().mul(-1)
                m_forward = torch.softmax(d, dim=1)
                m_backward = torch.softmax(d, dim=0)
                disc_loss = - args.rho * (cost * m_forward).sum(1).mean() - (1-args.rho) * (cost * m_backward).sum(0).mean()
            else:
                disc_loss = nn.BCEWithLogitsLoss()(discriminator(data), (torch.ones(args.batch_size, 1).cuda())) + \
                    nn.BCEWithLogitsLoss()(discriminator(generator(z)), (torch.zeros(args.batch_size, 1).cuda()))
            disc_loss.backward()
            optim_disc.step()

        # update generator
        for _ in range(args.gen_iters):
            z = (torch.randn(args.batch_size, Z_dim).cuda())
            optim_disc.zero_grad()
            optim_gen.zero_grad()
            if args.loss == 'hinge' or args.loss == 'wasserstein':
                gen_loss = -discriminator(generator(z)).mean()
            elif args.loss == 'ct':
                optim_nav.zero_grad()
                feat = discriminator(generator(z))
                feat_x = discriminator(data)
                mse_n = (feat_x[:,None] - feat).pow(2)
                cost = mse_n.sum(-1)
                d = navigator(mse_n).squeeze().mul(-1)
                m_forward = torch.softmax(d, dim=1)
                m_backward = torch.softmax(d, dim=0)
                gen_loss = args.rho * (cost * m_forward).sum(1).mean() + (1-args.rho) * (cost * m_backward).sum(0).mean()
            else:
                gen_loss = nn.BCEWithLogitsLoss()(discriminator(generator(z)), (torch.ones(args.batch_size, 1).cuda()))
            gen_loss.backward()
            if args.loss == 'ct':
                optim_nav.step()
            optim_gen.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}/{args.epochs}\tIt {batch_idx}/{len(loader)}\t" + 'disc loss', disc_loss.item(), 'gen loss', gen_loss.item())
            
    scheduler_d.step()
    scheduler_g.step()
    scheduler_n.step()

fixed_z = (torch.randn(args.batch_size, Z_dim).cuda())
def evaluate(epoch):

    samples = generator(fixed_z).cpu().data.numpy()[:64]


    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.transpose((1,2,0)) * 0.5 + 0.5)

    plt.savefig(os.path.join(args.out_dir,'{}.png'.format(str(epoch).zfill(3))), bbox_inches='tight')
    plt.close(fig)

os.makedirs(args.checkpoint_dir, exist_ok=True)
os.makedirs(args.out_dir, exist_ok=True)

for epoch in range(args.epochs):
    train(epoch)
    evaluate(epoch)
    torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(epoch)))
    torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(epoch)))

