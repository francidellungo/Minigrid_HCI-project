import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

from reward_nets.base_reward_net import RewardNet
from utils import *


class EmbMlpRewardNet(RewardNet):

    def __init__(self, input_shape, lr=1e-3, folder=None):
        # TODO sistemare signature di costruttore e init
        super(EmbMlpRewardNet, self).__init__(input_shape, lr, folder)
        self.input_shape = input_shape

        sizes = [6, 6, 1]
        self.embedding_ch0 = nn.Embedding(11, sizes[0])
        self.num_colors = sizes[1]
        self.embedding_ch2 = nn.Embedding(3, sizes[2])

        num_conv_filters = 100
        self.conv = nn.Conv2d(in_channels=sum(sizes), out_channels=num_conv_filters, kernel_size=1)

        num_hidden_units = 100
        self.affine1 = nn.Linear(num_conv_filters * np.prod(input_shape[1:]), num_hidden_units)
        self.affine2 = nn.Linear(num_hidden_units + 1, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=10 ** -4)
        # self.scheduler_kwargs = {"factor": 0.2, "patience": 10, "min_lr": 5 * 10 ** -5}
        # self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', **self.scheduler_kwargs)
        self.lambda_abs_rewards = 0#10 ** -4  # penalty for rewards regularization

    def forward(self, x, steps=None):
        x = x.view(-1, *self.input_shape).long()
        batch_size = len(x)

        steps = None
        if steps is None:
            steps = torch.zeros((batch_size, 1))
        assert len(steps) == batch_size

        # embedding/one hot
        e0 = self.embedding_ch0(x[:, 0, ...]).permute(0, 3, 1, 2)
        one_hot_ch1 = F.one_hot(x[:, 1, ...], self.num_colors).permute(0, 3, 1, 2).float()
        e2 = self.embedding_ch2(x[:, 2, ...]).permute(0, 3, 1, 2)

        x = torch.cat((e0, one_hot_ch1, e2), 1)
        x = F.relu(self.affine1(F.relu(self.conv(x).view(batch_size, -1))))
        s = steps.to(self.current_device()).float().view(batch_size, -1)

        return self.affine2(torch.cat((x, s), 1))


def get_net(*args, **kwargs):
    return EmbMlpRewardNet(*args, **kwargs)
