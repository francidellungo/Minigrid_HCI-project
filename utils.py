import os
import pickle
from glob import glob

import torch
from PyQt5.QtGui import QImage, QPixmap
from termcolor import colored
import numpy as np


def policies_dir():
    return "policy_nets"


def rewards_dir():
    return "reward_nets"


def games_dir():
    return "games"


# number of episodes for policy training
def num_max_episodes():
    return 10001


# numbers of episodes between each checkpoint
def get_episodes_for_checkpoint():
    return 100


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


def get_num_actions(env_name):
    if 'Empty' in env_name or 'FourRooms' in env_name:
        # print('env_name', env_name, ' ---> 3 ACTIONS')
        return 3
    # print('env_name', env_name, ' ---> 7 ACTIONS')
    return 7


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


def load_net(arg, eval_mode=False, device='auto'):
    if arg is None:
        return None
    if device == 'auto':
        device = auto_device()

    if arg.endswith(".pth"):
        # select specified weights
        checkpoint_to_load_weights = int(arg.rsplit("-", 1)[1].split(".", 1)[0])
        net_dir = os.path.dirname(arg)
        net = pickle.load(open(os.path.join(net_dir, "net.pkl"), "rb")).to(device)
        net.load_checkpoint(checkpoint_to_load_weights)

    else:
        # load the most recent weights from the specified folder
        net_dir = arg
        # net_dir = os.path.dirname(arg)
        net = pickle.load(open(os.path.join(net_dir, "net.pkl"), "rb")).to(device)
        net.load_last_checkpoint()

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


def standardize_with_memory(values, mem=0.9):
    running_avg = sum(values)/len(values)
    running_std = np.std(values)
    yield standardize(values)
    while True:
        running_avg = mem * running_avg + (1-mem) * sum(values)/len(values)
        running_std = mem * running_std + (1-mem) * np.std(values) + 10 ** -7
        yield [(v-running_avg)/running_std for v in values]


def rounded_list(iterator, digits=2):
    new_list = []
    for element in iterator:
        new_list.append(round(element, digits))
    return new_list


class Standardizer:

    def __init__(self, mem):
        self.mem = mem
        self.running_avg = None
        self.running_std = None

    def standardize(self, values):
        if self.running_avg is None:
            self.running_avg = np.mean(values)
            self.running_std = np.std(values) + 10 ** -7
            return [(v-self.running_avg)/self.running_std for v in values]
        self.running_avg = self.mem * self.running_avg + (1-self.mem) * sum(values)/len(values)
        self.running_std = self.mem * self.running_std + (1-self.mem) * np.std(values) + 10 ** -7
        return [(v-self.running_avg)/self.running_std for v in values]


class SumStandardizer:

    def __init__(self, history_length):
        self.history_length = history_length
        self.history = []

    def standardize(self, values):

        if len(self.history) == 0:  # first time: initialize
            self.history.append(sum(values))
            return values
        if len(self.history) == self.history_length:  # if history is full: remove the oldest
            self.history.pop(0)

        self.history.append(sum(values))
        avg = np.mean(self.history)
        std = np.std(self.history) + 0.1
        return [(v-(avg/len(values)))/std for v in values]
