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

        sizes = [6, 6, 1]
        self.embedding_ch0 = nn.Embedding(11, sizes[0])
        self.num_colors = sizes[1]
        self.embedding_ch2 = nn.Embedding(3, sizes[2])

        num_conv_filters = 100
        self.conv = nn.Conv2d(in_channels=sum(sizes), out_channels=num_conv_filters, kernel_size=1)

        num_hidden_units = 100
        self.affine1 = nn.Linear(num_conv_filters * np.prod(input_shape[1:]), num_hidden_units)
        self.affine2 = nn.Linear(num_hidden_units, num_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=10 ** -3)
        self.scheduler_kwargs = {"factor": 0.2, "patience": 10, "min_lr": 5 * 10 ** -5}
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', **self.scheduler_kwargs)

    def forward(self, x):
        x = x.view(-1, *self.input_shape).long()
        batch_size = len(x)

        # One hot encoding of each channel
        e0 = self.embedding_ch0(x[:, 0, ...]).permute(0, 3, 1, 2)
        one_hot_ch1 = F.one_hot(x[:, 1, ...], self.num_colors).permute(0, 3, 1, 2).float()
        e2 = self.embedding_ch2(x[:, 2, ...]).permute(0, 3, 1, 2)

        x = torch.cat((e0, one_hot_ch1, e2), 1)

        act_probs = self.affine2(F.relu(self.affine1(F.relu(self.conv(x).view(batch_size, -1)))))
        return act_probs


def get_net(*args, **kwargs):
    return MlpPolicyNet(*args, **kwargs)
