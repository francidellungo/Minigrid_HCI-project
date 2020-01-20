#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

import json
import os
import pickle
from datetime import datetime
import importlib
import argparse, argcomplete
from glob import glob

import gym
import gym_minigrid
import torch

from reward_nets.base_reward_net import RewardNet
from utils import get_all_environments, get_num_actions, get_input_shape, auto_device, load_reward

default_env = "MiniGrid-Empty-6x6-v0"
default_policy = "policy_nets/conv_policy.py"


def train_policy(env_name, policy_net=default_policy, reward_net_arg=None, policy_net_key=None, callbacks=[]):

    args_log = {"env_name": env_name, "policy": policy_net, "reward": reward_net_arg}

    env = gym.make(env_name)
    #num_actions = env.action_space.n

    # use GPU if available, otherwise use CPU
    device = auto_device()

    torch.manual_seed(0)  # for determinism, both on CPU and on GPU
    if torch.cuda.is_available():
        # required for determinism when using GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    module_path, _ = policy_net.rsplit(".", 1)
    file_radix = os.path.basename(os.path.normpath(module_path))
    net_module = importlib.import_module(".".join(module_path.split(os.sep)))

    policy_net_dir = module_path.rsplit("/", 1)[0] if "/" in module_path else ""  # TODO linux only
    if policy_net_key is None:
        policy_net_key = file_radix + "|" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    output_dir = os.path.join(policy_net_dir, env_name, policy_net_key)
    os.makedirs(output_dir)
    with open(os.path.join(output_dir, "args.json"), "wt") as file:
        json.dump(args_log, file)

    policy_net = net_module.get_net(get_input_shape(), get_num_actions(), env, policy_net_key, folder=output_dir).to(device)
    policy_net.fit(episodes=10000, reward_loader=lambda: load_reward(reward_net_arg, True, device), autosave=True,
                   episodes_for_checkpoint=250, reward_net_key=get_reward_key(reward_net_arg),
                   reward_net_games=get_reward_games(reward_net_arg), callbacks=callbacks)

    # # save trained policy_net net
    # torch.save(policy_net.state_dict(), options.output)


def get_reward_key(reward_net_path):
    if reward_net_path is not None:
        if reward_net_path.endswith(".pth"):
            return os.path.dirname(reward_net_path)
        else:
            return os.path.basename(os.path.normpath(reward_net_path))
    else:
        return None


def get_reward_games(reward_net_path):
    if reward_net_path is None:
        return None
    if reward_net_path.endswith(".pth"):
        reward_net_dir = os.path.dirname(reward_net_path)
    else:
        reward_net_dir = reward_net_path

    with open(os.path.join(reward_net_dir, "training.json")) as file:
        j = json.load(file)

    return j["games"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a policy net.')
    parser.add_argument("-e", "--env_name", help="gym environment to load", default=default_env, choices=get_all_environments())
    parser.add_argument("-p", "--policy", help="policy net to train", default=default_policy)
    parser.add_argument("-r", "--reward", help="reward network to load", default=None)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    train_policy(args.env_name, args.policy, args.reward)
