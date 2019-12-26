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


def load_last_policy(directory):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load the most recent weights from the specified folder
    epochs_saved_weights = [int(state.rsplit("-", 1)[1].split(".", 1)[0]) for state in glob(os.path.join(directory, "policy_net-*.pth"))]
    epoch_to_load_weights = max(epochs_saved_weights)

    policy_net = torch.load(os.path.join(directory, "net.pth"))
    policy_net.load_state_dict(torch.load(os.path.join(directory, "policy_net-" + str(epoch_to_load_weights) + ".pth")))
    policy_net = policy_net.to(device)
    return policy_net


def state_filter(net, state):
    return torch.from_numpy(state['image'][:, :, 0]).float().to(next(net.parameters()).device) # TODO considerare tutti i canali invece che solo il primo?
