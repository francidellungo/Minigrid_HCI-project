import io
import os
from abc import abstractmethod
import json
from contextlib import redirect_stdout

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.distributions.categorical import Categorical

from torchsummary import summary


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

    @abstractmethod
    def __init__(self, input_shape, num_actions, env):
        super(PolicyNet, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

    def fit(self, episodes=100, batch_size=5, reward=None, render=False, autosave=False, episodes_for_checkpoint=None, output_folder=""):

        print('output directory: "' + os.path.abspath(output_folder) + '"')
        tb_path = os.path.abspath(os.path.join(output_folder, "tensorboard"))
        print('to visualize training progress: `tensorboard --logdir="{}"`'.format(tb_path))
        self.save_training_details(output_folder, reward, batch_size)
        tensorboard = SummaryWriter(tb_path)

        torch.save(self, os.path.join(output_folder, "net.pth"))

        batch_loss = torch.zeros(1).to(next(self.parameters()).device)
        running_reward = None
        running_length = None
        for episode in range(episodes):

            self.optimizer.zero_grad()
            ep_loss = torch.zeros(1).to(next(self.parameters()).device)
            while ep_loss.item() == 0:
                states, actions, rewards, discounted_rewards, length = self.run_episode(max_length=100, reward_net=reward, render=render)

                for state, action, discounted_reward in zip(states[:-1], actions, discounted_rewards):
                    action_logits = self(state)
                    ep_loss += PolicyNet.loss(action_logits, action, discounted_reward)

            ep_loss /= len(actions)
            batch_loss += ep_loss

            if episode % batch_size == batch_size-1:
                batch_loss /= batch_size
                batch_loss.backward()
                self.optimizer.step()
                # TODO controllare loss: va in su e giù, capire se il problema sta qui oppure nella rete che dà il reward
                batch_loss = torch.zeros(1).to(next(self.parameters()).device)

            if autosave and episodes_for_checkpoint is not None and episode % episodes_for_checkpoint == 0:
                self.save_checkpoint(episode, output_folder)

            tot_ep_reward = sum(rewards)
            running_reward = tot_ep_reward if running_reward is None else running_reward * 0.95 + tot_ep_reward * 0.05
            running_length = length if running_length is None else running_length * 0.95 + length * 0.05

            print("episode: {},  episode_loss: {:7.4f},  length: {},  running_length: {:7.4f},  episode_reward: {:7.4f},  running_reward: {:7.4f}"
                  .format(episode, ep_loss.item(), length, running_length, tot_ep_reward, running_reward))

            tensorboard.add_scalars("loss", {"episode": ep_loss.item()}, episode)
            tensorboard.add_scalars("reward", {"episode": tot_ep_reward, "running": running_reward}, episode)
            tensorboard.add_scalars("length", {"episode": length, "running": running_length}, episode)

        tensorboard.close()
        if autosave and (episodes_for_checkpoint is None or (episodes-1) % batch_size != batch_size-1):  # save model if not saved in last episode
            self.save_checkpoint(episodes-1, output_folder)

    def state_filter(self, state):
        return torch.from_numpy(state['image'][:, :, 0]).float().to(next(self.parameters()).device)

    def sample_action(self, state):
        actions_logits = self(state)
        distribution = Categorical(logits=actions_logits)
        action = distribution.sample()
        return action

    def run_episode(self, max_length, reward_net=None, gamma=0.99, render=False):
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
            if reward_net is not None:
                with torch.no_grad():
                    reward = reward_net(state).item()  # TODO controllare nel paper quale è il modo giusto di calcolare il reward
            else:
                reward = r
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            if done:
                break

        discounted_rewards = PolicyNet.compute_discounted_rewards(rewards, gamma)

        return states, actions, rewards, discounted_rewards, step

    def save_training_details(self, output_folder, reward, batch_size):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        with open(os.path.join(output_folder, "training.json"), "wt") as file:
            r = "env" if reward is None else str(reward)
            # (7,7,3) == self.env.observation_space.spaces['image'].shape
            with io.StringIO() as out, redirect_stdout(out):
                summary(self, (1, 7, 7))
                net_summary = out.getvalue()
            print(net_summary)
            json.dump({"type": str(type(self)), "str": str(self).replace("\n", ""), "reward": r,
                       "batch_size": batch_size, "summary": net_summary}, file, indent=True)

    def save_checkpoint(self, episode, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        checkpoints_file = os.path.join(output_folder, "policy_net-" + str(episode) + ".pth")
        torch.save(self.state_dict(), checkpoints_file)
