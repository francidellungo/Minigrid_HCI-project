import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from policy_nets.base_policy_net import PolicyNet
from utils import conv_output_size, get_num_channels


class ConvPolicyNet(PolicyNet):

    def __init__(self, input_shape, num_actions, env, key=None, folder=None):
        # TODO sistemare signature di costruttore e init
        super(ConvPolicyNet, self).__init__(input_shape, num_actions, env, key, folder)
        num_embeddings = 4
        self.embedding = nn.Embedding(9, num_embeddings)
        self.conv = nn.Conv2d(in_channels=get_num_channels()*num_embeddings, out_channels=128, kernel_size=2)
        o = conv_output_size(input_shape[1], 2, 0, 1)
        self.fc = nn.Linear(128 * o * o, num_actions)
        self.input_shape = input_shape

        self.optimizer = optim.Adam(self.parameters(), lr=10**-5)

    def forward(self, x):
        x = x.view(-1, *self.input_shape)
        batch_size = len(x)
        x = self.embedding(x.long()).view(batch_size, -1, *self.input_shape[1:]).float()
        actions_logits = self.fc(F.relu(self.conv(x)).view(batch_size, -1))
        return actions_logits


def get_net(*args, **kwargs):
    return ConvPolicyNet(*args, **kwargs)
