import json
import os
from glob import glob
from optparse import OptionParser

import torch

from torchsummary import summary

from reward_net import RewardNet

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
    games_info = sorted([json.load(open(file, "r")) for file in glob(games_info_files)], key=lambda x: x["score"])
    X_train = [torch.Tensor(game_info["trajectory"]).to(device) for game_info in games_info]
    X_test = X_train
    #X_test = [X_train.pop(1), X_train.pop(2), X_train.pop()]

    reward_net = RewardNet(input_shape).to(device)

    print("summary")
    print(summary(reward_net, input_shape, device=device))

    # evaluate before training
    reward_net.evaluate(X_test)

    # training
    reward_net.fit(X_train, max_epochs=100)

    # evaluate after training
    reward_net.evaluate(X_test)

    with torch.no_grad():
        for trajectory in X_test:
            print("score: " + str(reward_net(trajectory).sum()))

    # save trained reward net
    torch.save(reward_net.state_dict(), "reward_net.pth")  # TODO specificare file output da argomenti
