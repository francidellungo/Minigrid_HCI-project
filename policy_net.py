import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions.categorical import Categorical

from utils import conv_output_size


class PolicyNet(nn.Module):

    @staticmethod
    def compute_discounted_rewards(rewards, gamma=0.99):
        discounted_rewards = []
        running = 0.0
        for r in reversed(rewards):
            running = r + gamma * running
            discounted_rewards.append(running)
        return list(reversed(discounted_rewards))

    @staticmethod
    def loss(actions_logits, action, discounted_reward):
        distribution = Categorical(logits=actions_logits)
        return (-distribution.log_prob(action) * discounted_reward).view(-1)

    def __init__(self, input_shape, num_actions, env, reward):
        super(PolicyNet, self).__init__()
        self.conv = nn.Conv2d(1, 15, 2)
        o = conv_output_size(input_shape[1], 2, 0, 1)
        self.fc = nn.Linear(15 * o * o, num_actions)
        self.input_shape = input_shape
        self.optimizer = optim.Adam(self.parameters(), 10 ** -4)
        self.env = env
        self.reward = reward

    def forward(self, x):
        x = x.view(-1, *self.input_shape)
        batch_size = len(x)
        actions_logits = self.fc(F.relu(self.conv(x)).view(batch_size, -1))
        return actions_logits

    def fit(self, episodes=1000, render=False):

        for episode in range(episodes):

            states, actions, discounted_rewards, length = self.run_episode(max_length=100, render=render)

            self.optimizer.zero_grad()
            l = torch.zeros(1).to(self.fc.weight.device)
            for state, action, discounted_reward in zip(states[:-1], actions, discounted_rewards):
                action_logits = self(state)
                l += PolicyNet.loss(action_logits, action, discounted_reward)

            l.backward()
            self.optimizer.step()
            # TODO controllare loss: va in su e giù, capire se il problema sta qui oppure nella rete che dà il reward
            print("episode:", episode, " length:", length, " avg_loss:", l.item())

    def state_filter(self, state):
        return torch.from_numpy(state['image'][:, :, 0]).float().to(self.fc.weight.device)

    def sample_action(self, state):
        actions_logits = self(state)
        distribution = Categorical(logits=actions_logits)
        action = distribution.sample()
        return action

    def run_episode(self, max_length, gamma=0.99, render=False):
        state = self.state_filter(self.env.reset())

        states = [state]
        actions = []
        rewards = []

        step = 0
        for step in range(max_length):
            if render:
                self.env.render()
            action = self.sample_action(state)

            state, r, done, _ = self.env.step(action)
            state = self.state_filter(state)
            with torch.no_grad():
                reward = self.reward(state)  # TODO controllare nel paper quale è il modo giusto di calcolare il reward
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            if done:
                break

        discounted_rewards = PolicyNet.compute_discounted_rewards(rewards, gamma)

        return states, actions, discounted_rewards, step
