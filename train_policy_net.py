import json
import os
from datetime import datetime
import importlib
import argparse
from glob import glob

import gym
import gym_minigrid
import torch

from reward_nets.base_reward_net import RewardNet

default_env = "MiniGrid-Empty-6x6-v0"
default_policy = "policy_nets/conv_policy/policy_net.py"


def train_policy(env_name, policy_net=default_policy, policy_net_key=None, reward_net_path=None):

    env = gym.make(env_name)
    input_shape = 1, 7, 7
    num_actions = env.action_space.n

    # use GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)  # for determinism, both on CPU and on GPU
    if torch.cuda.is_available():
        # required for determinism when using GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # load trained reward_net net if specified, otherwise env reward_net will be used
    if reward_net_path is not None:
        # # TODO load reward_net in the same way as policy_net
        # reward_net = RewardNet(input_shape)
        # reward_net.load_state_dict(torch.load(options.reward_net))
        # reward_net = reward_net.to(device)
        # reward_net.eval()  # disable network trainability
        
        if reward_net_path.endswith(".pth"):
            # select specified weights
            epoch_to_load_weights = reward_net_path.rsplit("-", 1)[1].split(".", 1)[0]
            reward_net_dir = os.path.dirname(reward_net_path)
        else:
            # load the most recent weights from the specified folder
            episodes_saved_weights = [int(state.rsplit("-", 1)[1].split(".", 1)[0]) for state in glob(os.path.join(reward_net_path, "reward_net-*.pth"))]
            epoch_to_load_weights = max(episodes_saved_weights)
            reward_net_dir = reward_net_path

        reward_net = torch.load(os.path.join(reward_net_dir, "net.pth"))
        reward_net.load_state_dict(torch.load(os.path.join(reward_net_dir, "reward_net-" + str(epoch_to_load_weights) + ".pth")))
        reward_net = reward_net.to(device)
    else:
        reward_net = None

    #policy_net = PolicyNet(input_shape, num_actions, env, reward_net).to(device)
    module_path, _ = policy_net.rsplit(".", 1)
    net_module = importlib.import_module(module_path.replace("/", "."))
    policy_net = net_module.get_net(input_shape, num_actions, env).to(device)

    policy_net_dir = module_path.rsplit("/", 1)[0] if "/" in module_path else ""
    if policy_net_key is None:
        policy_net_key = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    output_dir = os.path.join(policy_net_dir, env_name, policy_net_key)
    os.makedirs(output_dir)
    with open(os.path.join(output_dir, "args.json"), "wt") as file:
        # TODO fix args not defined
        json.dump(vars(args), file)
    policy_net.fit(episodes=10000, reward=reward_net, autosave=True, output_folder=output_dir, episodes_for_checkpoint=250)

    # # save trained policy_net net
    # torch.save(policy_net.state_dict(), options.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a policy net.')
    parser.add_argument("-e", "--env_name", help="gym environment to load", default=default_env)
    parser.add_argument("-p", "--policy", help="policy net to train", default=default_policy)
    parser.add_argument("-r", "--reward", help="reward network to load", default=None)
    args = parser.parse_args()
    train_policy(args.env_name, args.policy, args.reward)
