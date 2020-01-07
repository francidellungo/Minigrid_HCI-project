import os
from glob import glob

import torch


def conv_output_size(input_size, filter_size, padding=0, stride=1):
    # formula for output dimension:
    # O = (D -K +2P)/S + 1
    # where:
    #   D = input size (height/length)
    #   K = filter size
    #   P = padding
    #   S = stride
    return (input_size - filter_size + 2 * padding) // stride + 1


def get_last_policy_episode(directory):
    epochs_saved_weights = [int(state.rsplit("-", 1)[1].split(".", 1)[0]) for state in glob(os.path.join(directory, "policy_net-*.pth"))]
    if len(epochs_saved_weights) > 0:
        return max(epochs_saved_weights)
    return None


def load_last_policy(directory):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load the most recent weights from the specified folder
    epoch_to_load_weights = get_last_policy_episode(directory)

    policy_net = torch.load(os.path.join(directory, "net.pth"))
    if epoch_to_load_weights is not None:
        policy_net.load_state_dict(torch.load(os.path.join(directory, "policy_net-" + str(epoch_to_load_weights) + ".pth")))
    policy_net = policy_net.to(device)
    return policy_net


def state_filter(net, state):
    return torch.from_numpy(state['image'][:, :, 0]).float().to(next(net.parameters()).device) # TODO considerare tutti i canali invece che solo il primo?


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
