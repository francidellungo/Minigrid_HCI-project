import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

from policy_nets.base_policy_net import PolicyNet
from utils import conv_output_size, get_input_shape


class MlpPolicyNet(PolicyNet):

    def __init__(self, input_shape, num_actions, env, key=None, folder=None):
        # TODO sistemare signature di costruttore e init
        super(MlpPolicyNet, self).__init__(input_shape, num_actions, env, key, folder)
        self.shape_prod = int(np.prod(input_shape))
        self.affine1 = nn.Linear(self.shape_prod, 100)
        self.affine2 = nn.Linear(100, num_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=10 ** -4)

    def forward(self, x):
        x = x.view(-1, self.shape_prod)
        x = F.relu(self.affine1(x))
        act_probs = self.affine2(x).clamp(-1000.0, +1000.0)
        return act_probs


def get_net(*args, **kwargs):
    return MlpPolicyNet(*args, **kwargs)
