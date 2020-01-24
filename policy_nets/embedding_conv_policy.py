import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from policy_nets.base_policy_net import PolicyNet
from utils import conv_output_size, get_num_channels


class EmbConvPolicyNet(PolicyNet):

    def __init__(self, input_shape, num_actions, env, key=None, folder=None):
        # TODO sistemare signature di costruttore e init
        super(EmbConvPolicyNet, self).__init__(input_shape, num_actions, env, key, folder)
        num_embeddings = [3, 2, 1]
        self.embedding_ch0 = nn.Embedding(11, num_embeddings[0])
        self.embedding_ch1 = nn.Embedding(6, num_embeddings[1])
        self.embedding_ch2 = nn.Embedding(3, num_embeddings[2])
        self.conv = nn.Conv2d(in_channels=sum(num_embeddings), out_channels=128, kernel_size=2)
        o = conv_output_size(input_shape[1], 2, 0, 1)
        self.fc = nn.Linear(128 * o * o, num_actions)
        self.input_shape = input_shape

        self.optimizer = optim.Adam(self.parameters(), lr=10**-5)

    def forward(self, x):
        x = x.view(-1, *self.input_shape).long()
        batch_size = len(x)
        e0 = self.embedding_ch0(x[:, 0, ...]).permute(0, 3, 1, 2).float()
        e1 = self.embedding_ch1(x[:, 1, ...]).permute(0, 3, 1, 2).float()
        e2 = self.embedding_ch2(x[:, 2, ...]).permute(0, 3, 1, 2).float()
        x = torch.cat((e0, e1, e2), 1)
        actions_logits = self.fc(F.relu(self.conv(x)).view(batch_size, -1))
        return actions_logits


def get_net(*args, **kwargs):
    return EmbConvPolicyNet(*args, **kwargs)
