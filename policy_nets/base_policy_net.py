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

    def fit(self, episodes=100, batch_size=16, reward=None, render=False, autosave=False, episodes_for_checkpoint=None, output_folder="", reward_net_key=None):

        ''' print info to open output directory and to open tensorboard '''
        print('output directory:\n"' + os.path.abspath(output_folder) + '"')
        tb_path = os.path.abspath(os.path.join(output_folder, "tensorboard"))
        print('to visualize training progress:\n`tensorboard --logdir="{}"`'.format(tb_path))
        tensorboard = SummaryWriter(tb_path)

        ''' save info about this training in training.json, and also save the structure of the network '''
        self.save_training_details(output_folder, reward, batch_size, episodes, reward_net_key)
        torch.save(self, os.path.join(output_folder, "net.pth"))

        ''' init metrics '''
        # used metrics:  loss, return, true_return, length
        batch_avg_loss = torch.zeros(1).to(next(self.parameters()).device)
        batch_avg_return = 0
        batch_avg_true_return = 0
        batch_avg_length = 0
        # running metrics are useful to have a smoother plot
        running_batch_avg_loss = None
        running_batch_avg_return = None
        running_batch_avg_true_return = None
        running_batch_avg_length = None

        # clear all gradients
        self.optimizer.zero_grad()

        ''' begin training '''
        for episode in range(episodes):

            #ep_loss = torch.zeros(1).to(next(self.parameters()).device)
            episode_loss = torch.zeros(1).to(next(self.parameters()).device)
            while True:
                ''' run an episode '''
                states, actions, true_rewards, rewards, discounted_rewards, length = self.run_episode(max_length=100, reward_net=reward, render=render)

                gradient_not_zero = False

                # calculate the loss for each action
                for state, action, discounted_reward in zip(states[:-1], actions, discounted_rewards):
                    action_logits = self(state)
                    action_loss = PolicyNet.loss(action_logits, action, discounted_reward)

                    # if at least one action produce a loss != 0, I can consider valid this episode
                    if action_loss.item() != 0:
                        # update episode loss
                        episode_loss += action_loss
                        # this episode is valid (at least one action produced a gradient != 0)
                        gradient_not_zero = True

                # repeat episode if no action produced a loss !=0
                if gradient_not_zero:
                    break

            # update metrics
            #episode_loss /= len(actions) # TODO è corretto dividere la loss per il numero di azioni effettuate?
            batch_avg_loss += episode_loss
            batch_avg_return += sum(rewards)
            batch_avg_true_return += sum(true_rewards)
            batch_avg_length += length

            # if this is the last episode of the batch
            if episode % batch_size == batch_size-1:
                ''' backpropagation and optimizer step to update net weights '''
                batch_avg_loss /= batch_size
                batch_avg_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                ''' update, print and save metrics '''
                # calculate mean
                batch_avg_return /= batch_size
                batch_avg_true_return /= batch_size
                batch_avg_length /= batch_size
                # calculate running
                running_batch_avg_return = batch_avg_return if running_batch_avg_return is None else running_batch_avg_return * 0.95 + batch_avg_return * 0.05
                running_batch_avg_true_return = batch_avg_true_return if running_batch_avg_true_return is None else running_batch_avg_true_return * 0.95 + batch_avg_true_return * 0.05
                running_batch_avg_length = batch_avg_length if running_batch_avg_length is None else running_batch_avg_length * 0.95 + batch_avg_length * 0.05

                # print all metrics
                print("episode: {},  batch_avg_loss: {:7.4f},  batch_avg_length: {:7.4f},  running_batch_avg_length: {:7.4f},  batch_avg_return: {:7.4f},  running_batch_avg_return: {:7.4f},  batch_avg_true_return: {:7.4f},  running_batch_avg_true_return: {:7.4f}"
                      .format(episode, batch_avg_loss.item(), batch_avg_length, running_batch_avg_length, batch_avg_return, running_batch_avg_return, batch_avg_true_return, running_batch_avg_true_return))

                # save all metrics for tensorboard
                tensorboard.add_scalars("loss", {"batch_avg": batch_avg_loss.item()}, episode)
                tensorboard.add_scalars("return", {"batch_avg": batch_avg_return, "running_batch_avg": running_batch_avg_return}, episode)
                tensorboard.add_scalars("true_return", {"batch_avg": batch_avg_true_return, "running_batch_avg": running_batch_avg_true_return}, episode)
                tensorboard.add_scalars("length", {"batch_avg": batch_avg_length, "running_batch_avg": running_batch_avg_length}, episode)

                # re-init all metrics
                batch_avg_loss = torch.zeros(1).to(next(self.parameters()).device)
                batch_avg_return = 0
                batch_avg_true_return = 0
                batch_avg_length = 0

            # check if I have to save the net weights in this episode
            if autosave and episodes_for_checkpoint is not None and episode % episodes_for_checkpoint == episodes_for_checkpoint -1:
                # save net weights
                self.save_checkpoint(episode, output_folder)

        ''' training ended '''
        tensorboard.close()
        # save net weights only if they haven't been saved in the last episode
        if autosave and (episodes_for_checkpoint is None or (episodes-1) % episodes_for_checkpoint != episodes_for_checkpoint-1):
            self.save_checkpoint(episodes-1, output_folder)

    ''' transform environment observation into neural network input '''
    def state_filter(self, state):
        return torch.from_numpy(state['image'][:, :, 0]).float().to(next(self.parameters()).device) # TODO considerare tutti i canali invece che solo il primo?

    ''' sample an action from the distribution given by the policy net '''
    def sample_action(self, state):
        actions_logits = self(state)
        distribution = Categorical(logits=actions_logits)
        action = distribution.sample()
        return action

    ''' run an episode and return all relevant information '''
    def run_episode(self, max_length, reward_net=None, gamma=0.99, render=False):
        with torch.no_grad():  # for the reward net the gradient is not required
            state = self.state_filter(self.env.reset())

            states = [state]
            actions = []
            rewards = []
            true_rewards = []

            step = 0
            for step in range(max_length):
                if render:
                    self.env.render()
                action = self.sample_action(state)

                # execute action
                state, true_reward, done, _ = self.env.step(action)

                state = self.state_filter(state)

                # use reward from the reward net if it exists, otherwise use environment reward
                if reward_net is not None:
                    reward = reward_net(state).item()  # TODO controllare nel paper quale è il modo giusto di calcolare il reward
                else:
                    reward = true_reward

                states.append(state)
                true_rewards.append(true_reward)
                rewards.append(reward)
                actions.append(action)
                if done:
                    break

            discounted_rewards = PolicyNet.compute_discounted_rewards(rewards, gamma)

            return states, actions, true_rewards, rewards, discounted_rewards, step

    ''' save net and training details in training.json '''
    def save_training_details(self, output_folder, reward, batch_size, episodes, reward_net_key=None):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        with open(os.path.join(output_folder, "training.json"), "wt") as file:
            reward_type = "env" if reward is None else "net"
            # (7,7,3) == self.env.observation_space.spaces['image'].shape
            with io.StringIO() as out, redirect_stdout(out):
                summary(self, (1, 7, 7))
                net_summary = out.getvalue()
            print(net_summary)
            name = os.path.split(output_folder)[-1]
            j = {"name": name, "type": str(type(self)), "str": str(self).replace("\n", ""), "reward_type": reward_type,
                       "batch_size": batch_size, "max_episodes": episodes, "summary": net_summary}

            if reward_type == "net":
                j["reward_net_key"] = reward_net_key
                j["reward_net_details"] = str(reward)

            json.dump(j, file, indent=True)

    ''' save net weights (remark: only weights are saved here, not the network structure!) '''
    def save_checkpoint(self, episode, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        checkpoints_file = os.path.join(output_folder, "policy_net-" + str(episode) + ".pth")
        torch.save(self.state_dict(), checkpoints_file)
