import json
import os
from datetime import datetime
import importlib
from optparse import OptionParser
from glob import glob

import gym
import gym_minigrid
import torch

from reward_nets.base_reward_net import RewardNet


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default="MiniGrid-Empty-6x6-v0"
    )
    parser.add_option(
        "-p",
        "--policy",
        dest="policy_net_file",
        help="policy net to train",
        default="policy_nets/conv_policy/policy_net.py"
    )
    parser.add_option(
        "-r",
        "--reward",
        dest="reward_net",
        help="reward network to load",
        default=None
    )
    (options, args) = parser.parse_args()

    env = gym.make(options.env_name)
    input_shape = 1, 7, 7
    num_actions = env.action_space.n

    # use GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)  # for determinism, both on CPU and on GPU
    if torch.cuda.is_available():
        # required for determinism when using GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    reward_net = None
    # load trained reward net if specified, otherwise env reward will be used
    if options.reward_net is not None:
        # # TODO load reward in the same way as policy
        # reward_net = RewardNet(input_shape)
        # reward_net.load_state_dict(torch.load(options.reward_net))
        # reward_net = reward_net.to(device)
        # reward_net.eval()  # disable network trainability
        
        if options.reward_net.endswith(".pth"):
            # select specified weights
            epoch_to_load_weights = options.reward_net.rsplit("-", 1)[1].split(".", 1)[0]
            reward_net_dir = os.path.dirname(options.reward_net)
        else:
            # load the most recent weights from the specified folder
            episodes_saved_weights = [int(state.rsplit("-", 1)[1].split(".", 1)[0]) for state in glob(os.path.join(options.reward_net, "reward_net-*.pth"))]
            epoch_to_load_weights = max(episodes_saved_weights)
            reward_net_dir = options.reward_net

        reward_net = torch.load(os.path.join(reward_net_dir, "net.pth"))
        reward_net.load_state_dict(torch.load(os.path.join(reward_net_dir, "reward_net-" + str(epoch_to_load_weights) + ".pth")))
        reward_net = reward_net.to(device)

    #policy_net = PolicyNet(input_shape, num_actions, env, reward_net).to(device)
    module_path, _ = options.policy_net_file.rsplit(".", 1)
    net_module = importlib.import_module(module_path.replace("/", "."))
    policy_net = net_module.get_net(input_shape, num_actions, env).to(device)

    policy_net_dir = module_path.rsplit("/", 1)[0] if "/" in module_path else ""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    output_dir = os.path.join(policy_net_dir, options.env_name, timestamp)
    os.makedirs(output_dir)
    with open(os.path.join(output_dir, "args.json"), "wt") as file:
        json.dump(vars(options), file)
    policy_net.fit(episodes=10000, reward=reward_net, autosave=True, output_folder=output_dir, episodes_for_checkpoint=250)

    # # save trained policy net
    # torch.save(policy_net.state_dict(), options.output)
