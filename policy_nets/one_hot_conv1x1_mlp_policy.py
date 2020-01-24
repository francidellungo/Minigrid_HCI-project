import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

from policy_nets.base_policy_net import PolicyNet
from utils import *


class MlpPolicyNet(PolicyNet):

    def __init__(self, input_shape, num_actions, env, key=None, folder=None):
        # TODO sistemare signature di costruttore e init
        super(MlpPolicyNet, self).__init__(input_shape, num_actions, env, key, folder)
        self.one_hot_ch = [9, 6, 3]

        num_conv_filters = 100
        self.conv = nn.Conv2d(in_channels=sum(self.one_hot_ch), out_channels=num_conv_filters, kernel_size=1)

        num_hidden_units = 100
        self.affine1 = nn.Linear(num_conv_filters * np.prod(input_shape[1:]), num_hidden_units)
        self.affine2 = nn.Linear(num_hidden_units, num_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=10 ** -3)
        self.scheduler_kwargs = {"factor": 0.2, "patience": 10, "min_lr": 5 * 10 ** -5}
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', **self.scheduler_kwargs)

    def forward(self, x):
        x = x.view(-1, *self.input_shape)
        batch_size = len(x)

        # One hot encoding of each channel
        one_hot_0 = F.one_hot(x[:, 0, ...].long(), self.one_hot_ch[0]).permute(0, 3, 1, 2)
        one_hot_1 = F.one_hot(x[:, 1, ...].long(), self.one_hot_ch[1]).permute(0, 3, 1, 2)
        one_hot_2 = F.one_hot(x[:, 2, ...].long(), self.one_hot_ch[2]).permute(0, 3, 1, 2)
        one_hot_x = torch.cat((one_hot_0, one_hot_1, one_hot_2), 1).float()

        act_probs = self.affine2(F.relu(self.affine1(F.relu(self.conv(one_hot_x).view(batch_size, -1)))))
        return act_probs


def get_net(*args, **kwargs):
    return MlpPolicyNet(*args, **kwargs)
