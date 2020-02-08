import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import conv_output_size, get_num_channels

from reward_nets.base_reward_net import RewardNet


class ConvRewardNet(RewardNet):
    def __init__(self, input_shape, lr=1e-3, folder=None):
        super(ConvRewardNet, self).__init__(input_shape, lr, folder)
        self.input_shape = input_shape

        # simple net with: 2D convolutional layer -> activation layer -> fully connected layer
        self.conv = nn.Conv2d(in_channels=get_num_channels(), out_channels=64, kernel_size=2)
        o = conv_output_size(input_shape[1], 2, 0, 1)
        self.fc = nn.Linear(64 * o * o + 1, 1)
        self.step_weight = nn.Linear(1, 1)

        # regularization
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.lambda_abs_rewards = 10 ** -4  # penalty for rewards regularization

        # TODO tutti questi iperparametri dovrebbero essere presi come parametri in ingresso
        # TODO tutti questi iperparametri sono completamente ad occhio: vanno scelti per bene (ma in che modo?)
        # TODO la loro scelta dipende da varie cose, ad esempio se consideriamo o meno anche le traiettorie complete nel calolo della loss oppure solo quelle parziali

    def forward(self, x, steps=None):
        x = x.view(-1, *self.input_shape)
        batch_size = len(x)
        if steps is None:
            steps = torch.zeros((batch_size, 1))
        assert len(steps) == batch_size

        x = F.relu(self.conv(x)).view(batch_size, -1)
        s = steps.to(self.current_device()).float().view(batch_size, -1)

        return self.fc(torch.cat((x, s), 1))


def get_net(*args, **kwargs):
    return ConvRewardNet(*args, **kwargs)
