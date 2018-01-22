from __future__ import print_function

import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader

import nntree



parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--method', type=str, default='em-ma', choices=['em-ma','em-sgd','sgd'],metavar='N',
                    help='optimization method')
parser.add_argument('--optim', type=str, default='sgd', choices=['sgd','adam'],metavar='N',
                    help='optimizer')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=500, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=5)
        self.conv2 = nn.Conv2d(3, 6, kernel_size=5)
        self.fc1 = nn.Linear(6*16, 100)
        self.fc2 = nn.Linear(100, 1)
        self.gamma = 1

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 6*16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.sigmoid(self.gamma * x)


tree = nntree.mk_tree(4)
print(tree)

if args.cuda:
    model = nntree.nn_tree(10, Net, tree, cuda=True)
else:
    model = nntree.nn_tree(10, Net, tree, cuda=False)

def weights_init(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
        nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
        m.bias.data.normal_(0., 0.02)

for node_id in model.nodes:
    for m in model.nodes[node_id].modules():
        weights_init(m)

if args.optim == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
elif args.optim == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

import time
for epoch in range(1, args.epochs + 1):
    scheduler.step()
    s = time.time()
    if args.method == 'em-ma':
        model.fine_tune_em(epoch, train_loader, optimizer, 
                           gamma=1., log_interval = args.log_interval, m_step='ma',
                           test_loader = test_loader)
    elif args.method == 'em-sgd':
        model.fine_tune_em(epoch, train_loader, optimizer, 
                           gamma=1., log_interval = args.log_interval, m_step='sgd',
                           test_loader = test_loader)
    else:
        model.fine_tune_sgd(epoch, train_loader, optimizer, 
                            gamma=1, log_interval = args.log_interval,
                            test_loader = test_loader)
    print(time.time()-s)


