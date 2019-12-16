import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from policy_nets.base_policy_net import PolicyNet
from utils import conv_output_size


class MlpPolicyNet(PolicyNet):

    def __init__(self, input_shape, num_actions, env):
        # TODO sistemare signature di costruttore e init
        super(MlpPolicyNet, self).__init__(input_shape, num_actions, env)

        self.affine1 = nn.Linear(7*7, 100)
        self.affine2 = nn.Linear(100, num_actions)
        self.saved_log_probs = []
        self.rewards = []

        weight_decay = 10 ** -4  # penalty for net weights L2 regularization TODO vedere se con weight decay è meglio o peggio
        self.optimizer = optim.Adam(self.parameters(), lr=10 ** -4)
        self.env = env

    def forward(self, x):
        x = x.view(-1, 7 * 7)
        x = F.relu(self.affine1(x))
        act_probs = self.affine2(x).clamp(-1000.0, +1000.0)
        return act_probs


def get_net(*args, **kwargs):
    return MlpPolicyNet(*args, **kwargs)
