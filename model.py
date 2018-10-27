import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.fc23 = nn.Linear(fc2_units, fc2_units)
        self.fc34 = nn.Linear(fc2_units, fc2_units)
        self.bc1 = nn.BatchNorm1d(fc1_units, affine=False)
        self.bc2 = nn.BatchNorm1d(fc2_units, affine=False)
        self.ln = nn.LayerNorm(33)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        #print(state.size())
        x = F.leaky_relu(self.fc1((state)))
        #print(x.size())
        #x = self.bc1(x)
        x = F.leaky_relu(self.fc2(x))
        #x = F.leaky_relu(self.fc23(x))
        #x = F.leaky_relu(self.fc34(x))
        #x = self.bc2(x)
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        #self.fc3 = nn.Linear(fc2_units, action_size)
        self.fc23 = nn.Linear(fc2_units,fc2_units)
        #self.fc34 = nn.Linear(fc2_units,fc2_units)
        #self.fc3 = nn.Linear(fc2_units, 1)
        self.bc1 = nn.BatchNorm1d(fcs1_units, affine=False)
        self.bc2 = nn.BatchNorm1d(fc2_units, affine=False)
        self.ln = nn.LayerNorm(33)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    
    def discretize(self, ts, bins, global_min=None, global_max=None):        
        
        num_bins = bins

        min_value = ts.min()
        max_value = ts.max()
        if min_value == max_value:
            min_value = global_min
            max_value = global_max
        step = (max_value - min_value) / num_bins
        ts_bins = np.arange(min_value, max_value, step)
    
        ts_bins = bins

        inds = np.digitize(ts, ts_bins)
        #binned_ts = tuple(str(i - 1) for i in inds)
        binned_ts = inds*step + min_value
        return binned_ts 
    
    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""        
        #new_state = np.digitize(state,np.linspace(state.min(), state.max(),20))*20+state.min() #self.discretize(state,5.0,-10.0,10.0)
        #xs = F.relu(self.fcs1(new_state.cuda().float()))
        xs = F.leaky_relu(self.fcs1((state)))
        #xs = self.bc1(xs)
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc23(x))
        #x = F.leaky_relu(self.fc34(x))
        #x = self.bc2(x)
        return self.fc3(x)

    