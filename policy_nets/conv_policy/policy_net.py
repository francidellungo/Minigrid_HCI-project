import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from policy_nets.base_policy_net import PolicyNet
from utils import conv_output_size


class ConvPolicyNet(PolicyNet):

    def __init__(self, input_shape, num_actions, env, key=None):
        # TODO sistemare signature di costruttore e init
        super(ConvPolicyNet, self).__init__(input_shape, num_actions, env, key)
        self.conv = nn.Conv2d(1, 64, 2)
        o = conv_output_size(input_shape[1], 2, 0, 1)
        self.fc = nn.Linear(64 * o * o, num_actions)
        self.input_shape = input_shape

        weight_decay = 10 ** -4  # penalty for net weights L2 regularization
        self.optimizer = optim.Adam(self.parameters(), lr=10**-5, weight_decay=weight_decay)

    def forward(self, x):
        x = x.view(-1, *self.input_shape)
        batch_size = len(x)
        actions_logits = self.fc(F.relu(self.conv(x)).view(batch_size, -1))
        return actions_logits


def get_net(*args, **kwargs):
    return ConvPolicyNet(*args, **kwargs)
