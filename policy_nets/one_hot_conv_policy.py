import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from policy_nets.base_policy_net import PolicyNet
from utils import conv_output_size, get_num_channels


class OneHotConvPolicyNet(PolicyNet):

    def __init__(self, input_shape, num_actions, env, key=None, folder=None):
        self.n = 9
        # TODO sistemare signature di costruttore e init
        super(OneHotConvPolicyNet, self).__init__(input_shape, num_actions, env, key, folder)
        self.conv = nn.Conv2d(in_channels=get_num_channels()*self.n, out_channels=64, kernel_size=2)
        o = conv_output_size(input_shape[1], 2, 0, 1)
        self.fc = nn.Linear(64 * o * o, num_actions)
        self.input_shape = input_shape

        self.optimizer = optim.Adam(self.parameters(), lr=10**-5)

    def forward(self, x):
        x = x.view(-1, *self.input_shape)
        batch_size = len(x)
        one_hot_x = F.one_hot(x.long(), self.n).view(batch_size, -1, *self.input_shape[1:]).float()
        # print(one_hot_x)

        # one_hot_x = one_hot_x.view(-1, *self.input_shape)
        y = self.conv(one_hot_x)
        actions_logits = self.fc(F.relu(y).view(batch_size, -1))
        return actions_logits


def get_net(*args, **kwargs):
    return OneHotConvPolicyNet(*args, **kwargs)
