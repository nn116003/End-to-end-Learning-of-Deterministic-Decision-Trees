from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader

import itertools



def to_one_hot(y, n_dims=None):
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    if y.is_cuda:
        y_one_hot = y_one_hot.cuda()
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot



def mk_tree(depth, tree=None):
    if tree is None:
        tree = {0:[1,2]}
        depth = depth-1
    for d in range(depth):
        child_ids = []
        for node_id in tree:
            child_ids = child_ids + tree[node_id]
        leaf_ids = set(child_ids) - set(tree.keys())
        max_id = max(child_ids + tree.keys())
        for leaf_id in leaf_ids:
            tree[leaf_id] = [max_id + 1, max_id + 2]
            max_id += 2
    return tree
        
        


class nn_node(nn.Module):
    def __init__(self, split_model):
        
        super(nn_node, self).__init__()
        
        self.split_fcn = split_model()
        self.val = None

    def split_prob(self, x):
        # > 0.5 ->right, <= 0.5 -> left 
        self.val = self.split_fcn(x) # (batch_size, 1) 
        return self.val

    def prob_to(self, side):
        if side  > 0.5: # right
            return self.val
        else: # left
            return 1 - self.val 

class nn_leaf(nn.Module):
    def __init__(self, class_num, reg=True):

        super(nn_leaf, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(class_num))
        #self.weight = nn.Parameter(torch.zeros(class_num) + 1./class_num)
        if reg:
            self.weight.data = self.prob(softmax=True).data

        # use in m_step to calc moving avarage
        self.pi_n = 1. / class_num
        self.pi_d = 1

    def prob(self, softmax):
        if softmax:
            return F.softmax(self.weight, 0)
        else:
            return self.weight

class nn_tree(object):
    def __init__(self, class_num, split_model, tree ={0:[1,2]} ,
                 max_depth = None, min_samples_leaf=None, cuda=True):

        self.split_model = split_model
        self.tree = tree
        self.class_num = class_num
        self.cuda = cuda

        self.nodes = {} #nodes[node_id] = nn_node instance
        self.leaves = {} #leaf[node_id] = nn_leaf instance
        self.c2p = {}   #c2p[node_id] = node_id of parent
        for node_id in self.tree:
            self.nodes[node_id] = nn_node(split_model)
            if cuda:
                self.nodes[node_id].cuda()
            self.c2p[self.tree[node_id][0]] = node_id
            self.c2p[self.tree[node_id][1]] = node_id
            
        self.root_id = list(set(self.tree.keys()) - set(self.c2p.keys()))[0]

        leaf_ids = set(self.c2p.keys()) - set(self.tree.keys())
        for leaf_id in leaf_ids:
            self.leaves[leaf_id] = nn_leaf(class_num)
            if cuda:
                self.leaves[leaf_id].cuda()

    def parameters(self):
        params = []
        for node_id in self.nodes:
            params = itertools.chain(params, self.nodes[node_id].parameters())
        for node_id in self.leaves:
            params = itertools.chain(params, self.leaves[node_id].parameters())
        return params

    def set_train(self):
        for node_id in self.nodes:
            self.nodes[node_id].split_fcn.train()

    def set_eval(self):
        for node_id in self.nodes:
            self.nodes[node_id].split_fcn.eval()

    def set_gamma(self, gamma):
        for node_id in self.nodes:
            self.nodes[node_id].split_fcn.gamma = gamma

        
    def path_to(self, node_id):
        c_id = node_id
        path = [c_id]  # id seq to node_id
        split_sides = {} # split_sides[node_id] = 1:path right, 0:path left
        while True:
            if c_id not in self.c2p:
                break
            p_id = self.c2p[c_id]
            path.append(p_id)
            split_sides[p_id] = int(self.tree[p_id][1] == c_id)
            c_id = p_id

        path.reverse()
        return path, split_sides

    def update_nodes(self, x):
        for node_id in self.nodes:
            _ = self.nodes[node_id].split_prob(x)

    def update_nodes_to(self, x, leaf_ids):
        leaf_path = []
        for leaf_id in leaf_ids:
            path, _ = self.path_to(leaf_id)
            leaf_path.extend(leaf_path)
        use_nodes = set(leaf_path)
        for node_id in use_nodes:
            _ = self.nodes[node_id].split_prob(x)

    def prob_to(self, node_id):
        # Re: make sure to update nodes before exec prob_to
        path, split_sides = self.path_to(node_id)
        prob_var = 1
        for i in split_sides:
            prob_var = prob_var * self.nodes[i].prob_to(split_sides[i]).view(-1, 1)
        return prob_var # (batch_size, 1)
        
        
    def pred_with_all_leaves(self, x=None, softmax=False):
        if x is not None:
            self.update_nodes(x)

        output = 0
        for leaf_id in self.leaves:
            output += torch.mm(self.prob_to(leaf_id), self.leaves[leaf_id].prob(softmax).view(1, -1)) # (batch_size, class_num)

        return output

    def pred_with_one_leaf(self, x=None, softmax=False):
        if x is None:
            l = self.nodes[self.root_id].val.size()[0]
        else:
            l = x.size()[0]
        output = torch.zeros(l, self.class_num)
        if self.cuda:
            output = output.cuda()
        for i in range(l):
            p_id = self.root_id
            while True:
                if x is None:
                    p = self.nodes[p_id].val.data[i,:].cpu().numpy()
                else:
                    p = self.nodes[p_id].split_prob(x[i,:]).data.cpu().numpy()
                c_id = self.tree[p_id][int(p > 0.5)]
                if c_id in self.leaves:
                    output[i,:] =  self.leaves[c_id].prob(softmax).data
                    break
                p_id = c_id
        return Variable(output)

    def _e_step(self, y, softmax=False):

        exp_dict = {}
        l = y.size()[0]
        z = 0
        for leaf_id in self.leaves:
            exp_dict[leaf_id] = {}
            exp_dict[leaf_id]["prob_to"] = self.prob_to(leaf_id)
            #pred in a leaf
            exp_dict[leaf_id]["latent"] = torch.mm(
                exp_dict[leaf_id]["prob_to"], 
                self.leaves[leaf_id].prob(softmax).view(1, -1)
            )[[np.arange(l).astype(int), y]].view(l,1) # (batch_size, 1)
            z += exp_dict[leaf_id]["latent"]

        for leaf_id in self.leaves:
            exp_dict[leaf_id]["latent"] = exp_dict[leaf_id]["latent"] / (z + 1e-10)

        return exp_dict

    def _m_step_ma(self, y, exp_dict, optimizer, a=0.1):
        # update for pi
        for leaf_id in self.leaves:
            y_one_hot = to_one_hot(y, self.class_num)
            self.leaves[leaf_id].pi_n = ( (1 - a) * self.leaves[leaf_id].pi_n ) + \
                                        ( a * (exp_dict[leaf_id]["latent"] * y_one_hot).mean(dim=0) ).detach()
            self.leaves[leaf_id].pi_d = ( (1 - a) * self.leaves[leaf_id].pi_d ) + \
                                        ( a * exp_dict[leaf_id]["latent"].mean() ).detach()
            if self.leaves[leaf_id].pi_d.data.cpu().numpy() < 1e-10:
                pass
            else:
                self.leaves[leaf_id].weight.data = (self.leaves[leaf_id].pi_n /  (self.leaves[leaf_id].pi_d)).data
        # update nodes params
        loss = 0
        optimizer.zero_grad()
        for leaf_id in self.leaves:
            loss += (-torch.log(exp_dict[leaf_id]["prob_to"]+1e-10)*exp_dict[leaf_id]["latent"].detach()).mean() / len(self.leaves)
        loss.backward()
        optimizer.step()
        
        return loss

    def _m_step_sgd(self, y, exp_dict, optimizer):
        loss = 0
        optimizer.zero_grad()
        for leaf_id in self.leaves:
            loss += (
                (-torch.log(exp_dict[leaf_id]["prob_to"]+1e-10)  
                 - 100*(F.log_softmax(self.leaves[leaf_id].weight, dim=0)*to_one_hot(y, self.class_num)).sum(dim=1).view(-1,1) 
                   )
                      * exp_dict[leaf_id]["latent"].detach() ).mean() / len(self.leaves)
        loss.backward()
        optimizer.step()
        
        return loss

    def fine_tune_em(self, epoch, train_loader, optimizer, log_interval = 10, gamma=1, 
                     m_step = 'ma', test_loader = None, test_interval = 200):

        self.set_train()
        self.set_gamma(gamma)

        l = len(train_loader)
        correct_all = 0
        correct_one_leaf = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            # forward path
            self.update_nodes(data)

            # e step
            exp_dict = self._e_step(target, softmax=(m_step=='sgd'))

            # m step
            if m_step == 'ma':
                loss = self._m_step_ma(target, exp_dict, optimizer = optimizer)
            else:
                loss = self._m_step_sgd(target, exp_dict, optimizer = optimizer)


            #monitering
            output = self.pred_with_all_leaves(softmax=(m_step=='sgd'))
            pred = output.data.max(1, keepdim=True)[1] 
            correct_all += pred.eq(target.data.view_as(pred)).cpu().sum()
            output = self.pred_with_one_leaf(softmax=(m_step=='sgd'))
            pred = output.data.max(1, keepdim=True)[1] 
            correct_one_leaf += pred.eq(target.data.view_as(pred)).cpu().sum()

            if (batch_idx + 1) % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.6f}\tAll_Acc: {:.6f}\tone_leaf_Acc: {:.6f}'.format(
                    epoch, 
                    (batch_idx + 1) * len(data), 
                    len(train_loader.dataset),
                    100. * (batch_idx + 1) / len(train_loader), 
                    loss.data[0], 
                    float(correct_all) / (log_interval * len(data)),
                    float(correct_one_leaf) / (log_interval * len(data))
                )
                )
                correct_all = 0
                correct_one_leaf = 0

            # test
            if (test_loader is not None) and ( (batch_idx + 1) % test_interval == 0 ):
                self.test(test_loader, softmax=(m_step=='sgd'))
                self.set_train()



    def fine_tune_sgd(self, epoch, train_loader, optimizer, log_interval = 10, gamma=1,
                      test_loader = None, test_interval = 200):
        self.set_train()
        self.set_gamma(gamma)

        l = len(train_loader)
        correct_all = 0
        correct_one_leaf = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            optimizer.zero_grad()
            output = self.pred_with_all_leaves(data, softmax=True)
            loss = F.nll_loss(torch.log(output), target)
            loss.backward()
            optimizer.step()

            pred = output.data.max(1, keepdim=True)[1] 
            correct_all += pred.eq(target.data.view_as(pred)).cpu().sum()
            output = self.pred_with_one_leaf(softmax=True)
            pred = output.data.max(1, keepdim=True)[1] 
            correct_one_leaf += pred.eq(target.data.view_as(pred)).cpu().sum()

            if (batch_idx + 1) % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.6f}\tAcc: {:.6f}\tone_leaf_Acc: {:.6f}'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                    100. * (batch_idx + 1) / len(train_loader), loss.data[0], 
                    float(correct_all) / (log_interval * len(data)),
                    float(correct_one_leaf) / (log_interval * len(data))
                )
                )
                correct_all = 0
                correct_one_leaf = 0

            # test
            if (test_loader is not None) and ( (batch_idx + 1) % test_interval == 0 ):
                self.test(test_loader, softmax=True)
                self.set_train()
        
    def test(self, test_loader, softmax=True):
        self.set_eval()

        l = len(test_loader)
        correct_all = 0
        correct_one_leaf = 0
        loss_all = 0
        loss_one_leaf = 0

        for data, target in test_loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            output = self.pred_with_all_leaves(data, softmax=softmax)
            pred = output.data.max(1, keepdim=True)[1] 
            correct_all += pred.eq(target.data.view_as(pred)).cpu().sum()
            loss_all += F.nll_loss(torch.log(output), target).data

            output = self.pred_with_one_leaf(softmax=softmax)
            pred = output.data.max(1, keepdim=True)[1] 
            correct_one_leaf += pred.eq(target.data.view_as(pred)).cpu().sum()
            loss_one_leaf += F.nll_loss(torch.log(output), target).data

        
        print('Test: Loss: {:.6f}\tAcc: {:.6f}\tone_leaf_Loss: {:.6f}\tone_leaf_Acc: {:.6f}'.format(
            loss_all[0]/(l*len(data)),
            float(correct_all) / (l*len(data)),
            loss_one_leaf[0]/(l*len(data)),
            float(correct_one_leaf) / (l*len(data))
        ))
        
        return loss_all/l, loss_one_leaf/l
            
    
if __name__ == '__main__':

    class Net(nn.Module):
        def __init__(self, gamma = 1):
            super(Net, self).__init__()
            self.fc = nn.Linear(3, 1)
            self.gamma = gamma

        def forward(self, x):
            x = self.fc(x)
            return F.sigmoid(x * self.gamma)

    tree = nn_tree( 3, Net, tree = {0:[1, 2], 1:[3, 4]}, cuda=True)
    print(tree.path_to(2))
    print(tree.path_to(4))
    print(tree.path_to(0))


#    tmp_input = Variable(torch.Tensor([[0, 1, 2]]).cuda())
#    print(tree.pred_with_all_leaves(tmp_input))
    

    class mydata(Dataset):
        def __init__(self):
            data = np.zeros(900).reshape(300,3)
            data[:100, 0] = 1
            data[100:200, 1] = 1
            data[200:, 2] = 1
            data = data + np.random.normal(scale=0.1, size=900).reshape(300,3)
            
            target = np.zeros(300)
            target[100:200] = 1
            target[200:] = 2

            self.data = torch.from_numpy(data).float()
            self.target = torch.from_numpy(target).long()

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, i):
            return self.data[i,:], self.target[i]

    tmp_data = mydata()

    train_loader = DataLoader(tmp_data, batch_size=10, shuffle=True)        
    
    optimizer  = optim.SGD(tree.parameters(), lr=0.01, momentum=0.5)

    for leaf_id in tree.leaves:
        print(tree.leaves[leaf_id].weight)

    for i in range(10):
#        tree.fine_tune_sgd(i, train_loader, optimizer, gamma = 1*(1.5**i))
        tree.fine_tune_em(i, train_loader, optimizer, gamma=1*(1.5**i))

#    for node_id in tree.nodes:
#        print(tree.nodes[node_id].split_fcn.fc.weight)
    for leaf_id in tree.leaves:
        print(tree.leaves[leaf_id].weight)
