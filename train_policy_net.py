import gym
import torch

from policy_net import PolicyNet
from reward_net import RewardNet


if __name__ == "__main__":

    input_shape = 1, 7, 7
    num_actions = 7
    env = gym.make('MiniGrid-Empty-6x6-v0') # TODO scegliere ambiente da argomenti riga di comando

    # load trained reward net
    reward_net = RewardNet(input_shape)
    reward_net.load_state_dict(torch.load("reward_net.pth")) # TODO scegliere rete da argomenti
    reward_net.eval()  # disable network trainability

    policy_net = PolicyNet(input_shape, num_actions, env, reward_net)

    policy_net.fit(episodes=1000)

    # save trained policy net
    torch.save(policy_net.state_dict(), "policy_net.pth")
