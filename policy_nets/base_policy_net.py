import io
import os
import pickle
from abc import abstractmethod
import json
from contextlib import redirect_stdout
from glob import glob
from threading import Thread

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.distributions.categorical import Categorical

from modelsummary import summary

from utils import *


class PolicyNet(nn.Module):

    @staticmethod
    def compute_discounted_rewards(rewards, gamma=0.95):
        discounted_rewards = []
        running = 0.0
        for r in reversed(rewards):
            running = r + gamma * running
            discounted_rewards.append(running)
        return list(reversed(discounted_rewards))

    @staticmethod
    def loss(actions_logits, action, discounted_reward):
        distribution = Categorical(logits=actions_logits)
        return (-distribution.log_prob(action) * discounted_reward).view(-1)  # - (distribution.entropy() * (10 ** -2))  # TODO questo iperparametro va messo in un altro modo

    @abstractmethod
    def __init__(self, input_shape, num_actions, env, key=None, folder=None, episode_to_load=None):
        super(PolicyNet, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.env = env
        self.key = key
        self.episode = 0
        self.max_episodes = 0
        self.reward_net_key = None
        self.games = []
        self.name = os.path.basename(os.path.normpath(folder))
        if folder is None:
            folder = os.path.curdir
        self.folder = folder

        if episode_to_load is not None:
            self.load_checkpoint(episode_to_load)

        self.interrupted = False
        self.running = False
        self.standardizer = Standardizer(0.999)
        self.sum_standardizer = SumStandardizer(500)

    @abstractmethod
    def forward(self, x):
        pass

    def current_device(self):
        return next(self.parameters()).device

    def fit(self, episodes=100, batch_size=4, reward_loader=lambda:..., render=False, autosave=False, episodes_for_checkpoint=None, reward_net_key=None, reward_net_games=None, callbacks=[]):
        self.max_episodes = self.episode + episodes

        # TODO vedere se c'è verso prenderli dalla reward net invece che come parametro
        if reward_net_key is not None:
            self.reward_net_key = reward_net_key
        if reward_net_games is not None:
            self.games = reward_net_games

        for callback in callbacks:
            if "on_train_begin" in callback:
                callback["on_train_begin"](self)

        reward = reward_loader()
        # TODO check why CPU is very slow

        ''' print info to open output directory and to open tensorboard '''
        print('output directory:\n"' + os.path.abspath(self.folder) + '"')
        tb_path = os.path.abspath(os.path.join(self.folder, "tensorboard"))
        print('to visualize training progress:\n`tensorboard --logdir="{}"`'.format(tb_path))
        tensorboard = SummaryWriter(tb_path)

        ''' save info about this training in training.json, and also save the structure of the network '''
        self._save_training_details(reward, batch_size, self.reward_net_key)
        self.save_network()

        ''' init metrics '''
        # used metrics:  loss, return, true_return, length
        batch_avg_loss = torch.zeros(1).to(self.current_device())
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
        self.running = True
        for self.episode in range(self.episode, self.max_episodes):

            episode_loss = torch.zeros(1).to(self.current_device())
            while True:
                ''' run an episode '''
                states, actions, true_rewards, rewards, discounted_rewards, length = self.run_episode(max_length=self.env.max_steps, reward_net=reward, render=render)

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

            for callback in callbacks:
                if "before_update" in callback:
                    callback["before_update"](self)

            # if this is the last episode of the batch
            if self.episode % batch_size == batch_size-1:
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
                smooth_weight = 0.9
                running_batch_avg_return = batch_avg_return if running_batch_avg_return is None else running_batch_avg_return * smooth_weight + batch_avg_return * (1-smooth_weight)
                running_batch_avg_true_return = batch_avg_true_return if running_batch_avg_true_return is None else running_batch_avg_true_return * smooth_weight + batch_avg_true_return * (1-smooth_weight)
                running_batch_avg_length = batch_avg_length if running_batch_avg_length is None else running_batch_avg_length * smooth_weight + batch_avg_length * (1-smooth_weight)

                # print all metrics
                print("\repisode {}, avg_loss {:6.3f}, avg_length {:6.3f} (running {:6.3f}), avg_return {:6.3f} (running {:6.3f}), avg_true_return {:6.3f} (running {:6.3f}), lr {}   "
                      .format(self.episode, batch_avg_loss.item(), batch_avg_length, running_batch_avg_length, batch_avg_return, running_batch_avg_return, batch_avg_true_return, running_batch_avg_true_return, self.optimizer.param_groups[0]['lr']),
                      end="")

                # save all metrics for tensorboard
                tensorboard.add_scalars("loss", {"batch_avg": batch_avg_loss.item()}, self.episode)
                tensorboard.add_scalars("return", {"batch_avg": batch_avg_return, "running_batch_avg": running_batch_avg_return}, self.episode)
                tensorboard.add_scalars("true_return", {"batch_avg": batch_avg_true_return, "running_batch_avg": running_batch_avg_true_return}, self.episode)
                tensorboard.add_scalars("length", {"batch_avg": batch_avg_length, "running_batch_avg": running_batch_avg_length}, self.episode)

                if hasattr(self, "scheduler"):
                    self.scheduler.step(batch_avg_return)

                # re-init all metrics
                batch_avg_loss = torch.zeros(1).to(self.current_device())
                batch_avg_return = 0
                batch_avg_true_return = 0
                batch_avg_length = 0

            # check if I have to save the net weights in this episode
            if autosave:
                if self.episode == 0 or (episodes_for_checkpoint is not None and self.episode % episodes_for_checkpoint == 0) or self.episode==self.max_episodes-1:
                    # save net weights
                    self.save_checkpoint()

            for callback in callbacks:
                if "on_episode_end" in callback:
                    callback["on_episode_end"](self)

            if self.interrupted:
                break

        self.running = False

        ''' training ended '''
        tensorboard.close()

        for callback in callbacks:
            if "on_train_end" in callback:
                callback["on_train_end"](self)

    ''' sample an action from the distribution given by the policy net '''
    def sample_action(self, state):
        actions_logits = self(state)
        distribution = Categorical(logits=actions_logits)
        action = distribution.sample()
        return action

    ''' run an episode and return all relevant information '''
    def run_episode(self, max_length, reward_net=None, gamma=0.85, render=False):
        with torch.no_grad():  # for the reward net the gradient is not required
            state = state_filter(self.env.reset(), self.current_device())

            states = [state]
            actions = []
            rewards = []
            true_rewards = []

            step = 0
            for step in range(max_length):
                if render:
                    self.env.render()
                action = self.sample_action(state)

                #print_state(state)
                # execute action
                state, true_reward, done, _ = self.env.step(action)
                state = self.env.gen_obs()

                state = state_filter(state, self.current_device())

                # use reward from the reward net if it exists, otherwise use environment reward
                if reward_net is not None:
                    reward = reward_net(state, torch.tensor([self.env.step_count])).item()  # TODO controllare nel paper quale è il modo giusto di calcolare il reward
                else:
                    reward = true_reward

                states.append(state)
                true_rewards.append(true_reward)
                rewards.append(reward)
                actions.append(action)
                if done:
                    break

            #rewards = standardize(rewards)
            #rewards = normalize(rewards)
            #rewards = self.standardizer.standardize(rewards)
            #rewards = self.sum_standardizer.standardize(rewards)

            discounted_rewards = PolicyNet.compute_discounted_rewards(rewards, gamma)
            #discounted_rewards = standardize(PolicyNet.compute_discounted_rewards(rewards, gamma))
            #discounted_rewards = normalize(PolicyNet.compute_discounted_rewards(rewards, gamma))
            #discounted_rewards = self.sum_standardizer.standardize(discounted_rewards)

            return states, actions, true_rewards, rewards, discounted_rewards, step

    ''' save net and training details in training.json '''
    def _save_training_details(self, reward, size, reward_net_key=None):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        with open(os.path.join(self.folder, "training.json"), "wt") as file:
            reward_type = "env" if reward is None else "net"
            with io.StringIO() as out, redirect_stdout(out):
                summary(self, torch.zeros(get_input_shape()).to(self.current_device()), show_input=True)
                summary(self, torch.zeros(get_input_shape()).to(self.current_device()), show_input=False)
                net_summary = out.getvalue()
            print(net_summary)
            # self.name = os.path.basename(os.path.normpath(self.folder))
            j = {"name": self.name, "type": str(type(self)), "str": str(self).replace("\n", ""), "reward_type": reward_type,
                 "size": size, "max_episodes": self.max_episodes, "optimizer": str(self.optimizer),
                 "summary": net_summary}

            if hasattr(self, "scheduler"):
                j["scheduler"] = str(self.scheduler.__class__.__name__)
                if hasattr(self, "scheduler_kwargs"):
                    j["scheduler_kwargs"] = self.scheduler_kwargs

            if reward_type == "net":
                j["reward_net_key"] = reward_net_key
                j["reward_net_details"] = str(reward)

            json.dump(j, file, indent=True)

    def save_network(self):
        # torch.save(self, os.path.join(self.folder, "net.pth"))
        with open(os.path.join(self.folder, "net.pkl"), "wb") as file:
            pickle.dump(self, file)
        return self

    ''' save net weights (remark: only weights are saved here, not the network structure!) '''
    def save_checkpoint(self):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        checkpoints_file = os.path.join(self.folder, "policy_net-" + str(self.episode) + ".pth")
        torch.save(self.state_dict(), checkpoints_file)
        return self

    def load_checkpoint(self, episode_to_load_weights):
        # episode_to_load_weights = number of the episode
        if episode_to_load_weights is not None:
            self.load_state_dict(
                torch.load(os.path.join(self.folder, "policy_net-" + str(episode_to_load_weights) + ".pth"),
                           map_location=self.current_device()))

            self.episode = episode_to_load_weights
            self.max_episodes = self._get_last_training_max_episodes()
        else:
            print("Error: cannot find saved weights")
        return self

    def load_last_checkpoint(self):
        # load the most recent weights from the folder of this policy
        episode_to_load_weights = self._get_last_saved_policy_episode()
        return self.load_checkpoint(episode_to_load_weights)

    def _get_last_saved_policy_episode(self):
        episodes_saved_weights = [int(state.rsplit("-", 1)[1].split(".", 1)[0]) for state in glob(os.path.join(self.folder, "policy_net-*.pth"))]
        if len(episodes_saved_weights) > 0:
            return max(episodes_saved_weights)
        return None

    def _get_last_training_max_episodes(self):
        with open(os.path.join(self.folder, "training.json"), "rt") as file:
            j = json.load(file)
        return j["max_episodes"]

    def interrupt(self):
        self.interrupted = True
        self.running = True

    def play(self):
        self.running = True

    def pause(self):
        self.running = False
