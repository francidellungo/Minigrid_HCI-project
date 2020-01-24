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
        # self.shape_prod = np.prod(input_shape)
        # self.input_shape_lin = [1, sum((9, 6, 3)), 7, 7]
        self.affine1 = nn.Linear(sum(self.one_hot_ch) * 7 * 7, 100)
        self.affine2 = nn.Linear(100, num_actions)
        self.saved_log_probs = []
        self.rewards = []

        self.optimizer = optim.Adam(self.parameters(), lr=10 ** -4)
        self.scheduler_kwargs = {"factor": 0.2, "patience": 10, "min_lr": 10 ** -4}
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', **self.scheduler_kwargs)

    def forward(self, x):
        # print('state: ', x)
        x = x.view(-1, *self.input_shape)
        batch_size = len(x)

        # One hot encoding of each channel
        one_hot_0 = F.one_hot(x[:, 0, ...].long(), self.one_hot_ch[0]).permute(0, 3, 1, 2)
        one_hot_1 = F.one_hot(x[:, 1, ...].long(), self.one_hot_ch[1]).permute(0, 3, 1, 2)
        one_hot_2 = F.one_hot(x[:, 2, ...].long(), self.one_hot_ch[2]).permute(0, 3, 1, 2)
        one_hot_x = torch.cat((one_hot_0, one_hot_1, one_hot_2), 1).float().view(batch_size, -1)

        act_probs = self.affine2(F.relu(self.affine1(one_hot_x))).clamp(-1000.0, +1000.0)
        return act_probs


def get_net(*args, **kwargs):
    return MlpPolicyNet(*args, **kwargs)
