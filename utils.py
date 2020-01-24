import os
import pickle
from glob import glob

import torch
from PyQt5.QtGui import QImage, QPixmap
from termcolor import colored
import numpy as np


def games_dir():
    return "games"


def conv_output_size(input_size, filter_size, padding=0, stride=1):
    # formula for output dimension:
    # O = (D -K +2P)/S + 1
    # where:
    #   D = input size (height/length)
    #   K = filter size
    #   P = padding
    #   S = stride
    return (input_size - filter_size + 2 * padding) // stride + 1


def get_input_shape():
    return 3, 7, 7


def get_num_channels():
    return get_input_shape()[0]


def state_filter(obs, device='auto'):
    """
    :param device: device where to put the returned torch tensor
    :param obs: environment observation
    :return: torch 7x7x3 tensor
    """

    if device == 'auto':
        device = auto_device()

    obs_image = obs['image'].astype(float)
    #obs_image[obs_image == 6] = -1
    return torch.from_numpy(obs_image).float().permute(2, 0, 1).to(device)


def print_observation(obs, flip=True):
    colors = ["red", "green", "blue"]
    for i, color in enumerate(colors):
        obs_channel_i = obs['image'][:, :, i]
        if flip:
            obs_channel_i = np.flip(obs_channel_i, axis=1)
        #obs_channel_i = obs_channel_i.astype(float)
        #obs_channel_i[obs_channel_i == 6] = -1
        print(colored(obs_channel_i, color))


def print_state(state, flip=True):
    state = state.to("cpu")
    colors = ["red", "green", "blue"]
    for i, color in enumerate(colors):
        obs_channel_i = state[i, ...]
        # if flip:
        #     obs_channel_i = np.flip(obs_channel_i, axis=1)
        print(colored(obs_channel_i, color))


def get_num_actions():
    return 3


def get_all_environments():
    return ["MiniGrid-Empty-5x5-v0", "MiniGrid-Empty-Random-5x5-v0", "MiniGrid-Empty-6x6-v0", "MiniGrid-Empty-Random-6x6-v0",
            "MiniGrid-Empty-8x8-v0", "MiniGrid-Empty-16x16-v0", "MiniGrid-FourRooms-v0", "MiniGrid-DoorKey-5x5-v0", "MiniGrid-DoorKey-6x6-v0",
            "MiniGrid-DoorKey-8x8-v0", "MiniGrid-DoorKey-16x16-v0", "MiniGrid-MultiRoom-N2-S4-v0", "MiniGrid-MultiRoom-N4-S5-v0",
            "MiniGrid-MultiRoom-N6-v0", "MiniGrid-Dynamic-Obstacles-5x5-v0", "MiniGrid-Dynamic-Obstacles-Random-5x5-v0", "MiniGrid-Dynamic-Obstacles-6x6-v0",
            "MiniGrid-Dynamic-Obstacles-Random-6x6-v0", "MiniGrid-Dynamic-Obstacles-8x8-v0", "MiniGrid-Dynamic-Obstacles-16x16-v0"]

    # return ["MiniGrid-Empty-5x5-v0", "MiniGrid-Empty-Random-5x5-v0", "MiniGrid-Empty-6x6-v0", "MiniGrid-Empty-Random-6x6-v0",
    #         "MiniGrid-Empty-8x8-v0", "MiniGrid-Empty-16x16-v0", "MiniGrid-FourRooms-v0", "MiniGrid-DoorKey-5x5-v0", "MiniGrid-DoorKey-6x6-v0",
    #         "MiniGrid-DoorKey-8x8-v0", "MiniGrid-DoorKey-16x16-v0", "MiniGrid-MultiRoom-N2-S4-v0", "MiniGrid-MultiRoom-N4-S5-v0",
    #         "MiniGrid-MultiRoom-N6-v0", "MiniGrid-Fetch-5x5-N2-v0", "MiniGrid-Fetch-6x6-N2-v0", "MiniGrid-Fetch-8x8-N3-v0",
    #         "MiniGrid-GoToDoor-5x5-v0", "MiniGrid-GoToDoor-7x7-v0", "MiniGrid-GoToDoor-8x8-v0", "MiniGrid-PutNear-6x6-N2-v0",
    #         "MiniGrid-PutNear-8x8-N3-v0", "MiniGrid-RedBlueDoors-6x6-v0", "MiniGrid-RedBlueDoors-8x8-v0", "MiniGrid-MemoryS17Random-v0",
    #         "MiniGrid-MemoryS13Random-v0", "MiniGrid-MemoryS13-v0", "MiniGrid-MemoryS11-v0", "MiniGrid-MemoryS9-v0", "MiniGrid-MemoryS7-v0",
    #         "MiniGrid-LockedRoom-v0", "MiniGrid-KeyCorridorS3R1-v0", "MiniGrid-KeyCorridorS3R2-v0", "MiniGrid-KeyCorridorS3R3-v0",
    #         "MiniGrid-KeyCorridorS4R3-v0", "MiniGrid-KeyCorridorS5R3-v0", "MiniGrid-KeyCorridorS6R3-v0", "MiniGrid-Unlock-v0",
    #         "MiniGrid-UnlockPickup-v0", "MiniGrid-BlockedUnlockPickup-v0", "MiniGrid-ObstructedMaze-1Dl-v0", "MiniGrid-ObstructedMaze-1Dlh-v0",
    #         "MiniGrid-ObstructedMaze-1Dlhb-v0", "MiniGrid-ObstructedMaze-2Dl-v0", "MiniGrid-ObstructedMaze-2Dlh-v0", "MiniGrid-ObstructedMaze-2Dlhb-v0",
    #         "MiniGrid-ObstructedMaze-1Q-v0", "MiniGrid-ObstructedMaze-2Q-v0", "MiniGrid-ObstructedMaze-Full-v0", "MiniGrid-DistShift1-v0",
    #         "MiniGrid-DistShift2-v0", "MiniGrid-LavaGapS5-v0", "MiniGrid-LavaGapS6-v0", "MiniGrid-LavaGapS7-v0", "MiniGrid-LavaCrossingS9N1-v0",
    #         "MiniGrid-LavaCrossingS9N2-v0", "MiniGrid-LavaCrossingS9N3-v0", "MiniGrid-LavaCrossingS11N5-v0", "MiniGrid-SimpleCrossingS9N1-v0",
    #         "MiniGrid-SimpleCrossingS9N2-v0", "MiniGrid-SimpleCrossingS9N3-v0", "MiniGrid-SimpleCrossingS11N5-v0", "MiniGrid-Dynamic-Obstacles-5x5-v0",
    #         "MiniGrid-Dynamic-Obstacles-Random-5x5-v0", "MiniGrid-Dynamic-Obstacles-6x6-v0", "MiniGrid-Dynamic-Obstacles-Random-6x6-v0",
    #         "MiniGrid-Dynamic-Obstacles-8x8-v0", "MiniGrid-Dynamic-Obstacles-16x16-v0"]


def auto_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_policy(policy_arg, eval_mode=False, device='auto'):
    return _load_net(policy_arg, "policy_net-", eval_mode, device)


def load_reward(reward_arg, eval_mode=False, device='auto'):
    return _load_net(reward_arg, "reward_net-", eval_mode, device)


def _load_net(arg, prefix, eval_mode=False, device='auto'):
    if arg is None:
        return None
    if device == 'auto':
        device = auto_device()

    if arg.endswith(".pth"):
        # select specified weights
        checkpoint_to_load_weights = arg.rsplit("-", 1)[1].split(".", 1)[0]
        net_dir = os.path.dirname(arg)

    else:
        # load the most recent weights from the specified folder
        checkpoints_saved_weights = [int(state.rsplit("-", 1)[1].split(".", 1)[0]) for state in glob(os.path.join(arg, prefix + "*.pth"))]
        checkpoint_to_load_weights = max(checkpoints_saved_weights)
        net_dir = arg

    net = pickle.load(open(os.path.join(net_dir, "net.pkl"), "rb")).to(device)
    net.load_state_dict(torch.load(os.path.join(net_dir, prefix + str(checkpoint_to_load_weights) + ".pth"), map_location=device))
    if eval_mode:
        net.eval()
    return net


def nparray_to_qpixmap(img):
    return QPixmap(QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888))


def normalize(values, inf=-1, sup=1):
    assert inf < sup
    mn = min(values)
    mx = max(values)
    ampl = (mx - mn + 10 ** -7)
    return [(v-mn)/ampl * (sup-inf) + inf for v in values]


def standardize(values):
    mean = np.mean(values)
    std = np.std(values) + 10 ** -7
    return [(v-mean)/std for v in values]


def rounded_list(iterator, digits=2):
    new_list = []
    for element in iterator:
        new_list.append(round(element, digits))
    return new_list
