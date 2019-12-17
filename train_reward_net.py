from datetime import datetime
import importlib
import json
import os
from glob import glob
from optparse import OptionParser

import torch

from torchsummary import summary

from reward_nets.base_reward_net import RewardNet

if __name__ == "__main__":

    games_path = 'games'

    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-Empty-6x6-v0'
    )
    parser.add_option(
        "-r",
        "--reward",
        dest="reward_net_file",
        help="reward network to train",
        default="reward_nets/conv_reward/reward_net.py"
    )
    (options, args) = parser.parse_args()

    # use GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)    # for determinism, both on CPU and on GPU
    if torch.cuda.is_available():
        # required for determinism when using GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ''' read all trajectories and create the training set as the set of trajectories ordered by score '''
    input_shape = 1, 7, 7
    games_path = os.path.join(games_path, options.env_name)
    games_directories = os.path.join(games_path, "*")
    games_info_files = os.path.join(games_directories, "game.json")

    list_game_info_files = glob(games_info_files)
    l = len(list_game_info_files)
    perm = torch.randperm(l)
    train_fraction = 2/3
    train_size = int(l * train_fraction)
    list_train_game_info_files = [f for f in list_game_info_files[:train_size]]
    list_val_game_info_files = [f for f in list_game_info_files[train_size:]]

    train_games_info = sorted([json.load(open(file, "r")) for file in list_train_game_info_files], key=lambda x: x["score"])
    val_games_info = sorted([json.load(open(file, "r")) for file in list_val_game_info_files], key=lambda x: x["score"])

    X_train = [torch.Tensor(game_info["trajectory"]).to(device) for game_info in train_games_info]
    X_val = [torch.Tensor(game_info["trajectory"]).to(device) for game_info in val_games_info]
    X_test = X_val

    #reward_net = RewardNet(input_shape).to(device)
    module_path, _ = options.reward_net_file.rsplit(".", 1)
    net_module = importlib.import_module(module_path.replace("/", "."))
    reward_net = net_module.get_net(input_shape).to(device)

    print("summary")
    summary(reward_net, input_shape, device=device)

    # evaluate before training
    #reward_net.evaluate(X_test, [reward_net.quality])

    # training
    reward_net_dir = module_path.rsplit("/", 1)[0] if "/" in module_path else ""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    output_dir = os.path.join(reward_net_dir, options.env_name, timestamp)
    reward_net.fit(X_train, max_epochs=20, X_val=X_val, output_folder=output_dir, train_games_info=train_games_info, val_games_info=val_games_info, autosave=True, epochs_for_checkpoint=10)

    # evaluate after training
    #reward_net.evaluate(X_test, [reward_net.quality])

    with torch.no_grad():
        for trajectory in X_test:
            print("score: " + str(reward_net(trajectory).sum()))

    # # save trained reward net
    # torch.save(reward_net.state_dict(), "reward_net.pth")
