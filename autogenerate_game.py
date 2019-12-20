import time
import gym
import gym_minigrid
import numpy as np
import cv2
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.distributions.categorical import Categorical
from itertools import count
from datetime import datetime
from plot_rewards import plot_reward

games_path = 'games'
plot_path = 'figures'

# A simple, memoryless MLP agent. Last layer are logits (scores for
# which higher values represent preferred actions.
class Policy(nn.Module):
    def __init__(self, obs_size, act_size, inner_size, **kwargs):
        super(Policy, self).__init__(**kwargs)
        self.affine1 = nn.Linear(obs_size, inner_size)
        self.affine2 = nn.Linear(inner_size, act_size)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = x.view(-1, 7*7)
        x = F.relu(self.affine1(x))
        act_probs = self.affine2(x).clamp(-1000.0, +1000.0)
        return act_probs

# Function that, given a policy network and a state selects a random
# action according to the probabilities output by final layer.
def select_action(policy, state):
    probs = policy.forward(state)
    dist = Categorical(logits=probs)
    action = dist.sample()
    return action

# Utility function.
# The MiniGrid gym environment uses 3 channels as
# state, but for this we only use the first channel: represents all
# objects (including goal) with integers. This function just strips
# out the first channel and returns it.
def state_filter(state):
    return torch.from_numpy(state['image'][:,:,0]).float()


# Function to compute discounted rewards after a complete episode.
def compute_discounted_rewards(rewards, gamma=0.99):
    discounted_rewards = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        discounted_rewards.append(running)
    return list(reversed(discounted_rewards))


# The function that runs the simulation for a specified length. The
# nice thing about the MiniGrid environment is that the game never
# ends. After achieving the goal, the game resets.
def run_episode(env, policy, length, save, gamma=0.99):
    """
    :param env: gym-minigrid environment used
    :param policy:
    :param length: max length of the episode
    :param save: save or not current episode
    :param gamma: parameter for calculation of discounted rewards
    :return:
    """
    # create dictionary to save game info
    trajectory = []
    # Restart the MiniGrid environment.
    state = state_filter(env.reset())

    # We need to keep a record of states, actions, and the
    # instantaneous rewards.
    states = [state]
    actions = []
    rewards = []

    screenshots = []
    # Run for desired episode length.
    for step in range(length):
        # Get action from policy net based on current state.
        action = select_action(policy, state)

        # Simulate one step, get new state and instantaneous reward.
        state, reward, done, _ = env.step(action)
        env.render()

        # Save state
        trajectory.append(state_filter(state).tolist())

        # print(step, reward, done)

        # Save screenshots
        screenshot_name = 'game' + str(env.step_count) + '.png'
        pixmap = env.render('pixmap')
        screenshots.append((screenshot_name, pixmap))

        state = state_filter(state)
        # print(step, state)
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        if done:
            break

    # Finished with episode, compute loss per step.
    discounted_rewards = compute_discounted_rewards(rewards, gamma)
    avg_rewards = np.mean(rewards)
    # print("discounted_rewards: ", discounted_rewards)

    # Return the sequence of states, actions, and the corresponding rewards.
    return (states, actions, discounted_rewards, avg_rewards, trajectory, screenshots)

###### The main loop.


if __name__ == '__main__':
    # Some configuration variables.
    episode_len = 50  # Length of each game.
    obs_size = 7*7    # MiniGrid uses a 7x7 window of visibility.
    act_size = 7      # Seven possible actions (turn left, right, forward, pickup, drop, etc.)
    inner_size = 64   # Number of neurons in two hidden layers.
    lr = 0.001        # Adam learning rate
    avg_discounted_reward = 0.0  # For tracking average regard per episode.
    avg_reward = 0.0  # avg reward per game
    rewards = []

    # Plot (matplotlib) and save rewards
    plot = True
    save_rewards_plot = False

    # Setup OpenAI Gym environment for guessing game.
    env = gym.make('MiniGrid-Empty-6x6-v0')
    # env = gym.make('MiniGrid-MultiRoom-N6-v0')
    env_name = 'MiniGrid-Empty-6x6-v0'

    # Instantiate a policy network.
    policy = Policy(obs_size=obs_size, act_size=act_size, inner_size=inner_size)

    # Use the Adam optimizer.
    optimizer = torch.optim.Adam(params=policy.parameters(), lr=lr)

    # Run for a while.
    episodes = 501

    # save 1 of num_save episodes
    num_save = 25

    for step in range(episodes):
        save_game = False

        # define if the current game has to be saved or not
        if step % num_save == 0 and step != 0:
            save_game = True

        # MiniGrid has a QT5 renderer which is pretty cool.
        env.render('human')
        # env.render('ansi')

        # define new game info
        game_name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        # print('New game: ', game_name)
        game_info = {
            'name': game_name,
            'trajectory': [],
            'score': None
        }

        time.sleep(0.01)

        # Run an episode.
        #TODO forse basta ritornare states che è la lista degli stati filtrati dell' ultima partita dell' episodio fatto
        (states, actions, discounted_rewards, avg_rewards, _, screenshots) = run_episode(env, policy, episode_len, save_game)
        avg_discounted_reward += np.mean(discounted_rewards)
        avg_reward += avg_rewards
        #TODO rewards and avg rewards...

        if save_game:
            print('New game: ', game_name)
            game_directory = os.path.join(games_path, env_name, str(game_name))
            game_info['trajectory'] = [state.tolist() for state in states]
            game_info['score'] = (avg_reward / num_save)
            print('Average reward @ episode {}: {}'.format(step, avg_reward / num_save))
            rewards.append((step, avg_reward / num_save))

            avg_discounted_reward = 0.0
            avg_reward = 0

        # Repeat each action, and backpropagate discounted rewards.
        # This can probably be batched for efficiency with a
        # memoryless agent...
        optimizer.zero_grad()
        for (step_, a) in enumerate(actions):
            logits = policy(states[step_])
            dist = Categorical(logits=logits)
            loss = -dist.log_prob(actions[step_]) * discounted_rewards[step_]
            loss.backward()
        optimizer.step()

        # save trained net
        if save_game:
            torch.save(policy.state_dict(), 'policy_state.pth')
            print('save')

            # save .json file
            # with open(str(game_info['name']) + '.json', 'w+') as game_file:
            #     json.dump(game_info, game_file, ensure_ascii=False)
            # Create new folder to save images and json
            if not os.path.exists(game_directory):
                os.makedirs(game_directory)

            # Save image of each state
            for screenshot_name, pixmap in screenshots:
                screenshot_path = os.path.join(game_directory, screenshot_name)
                pixmap.save(screenshot_path)

            with open(os.path.join(game_directory, 'game.json'), 'w+') as game_file:
                json.dump(game_info, game_file, ensure_ascii=False)

    # plot and save rewards
    if plot:
        plot_reward(rewards, save_rewards_plot, plot_path)

    # save file of avg scores
    with open(os.path.join(games_path, env_name, 'avg_rewards.json'), 'w+') as reward_file:
        json.dump(rewards, reward_file, ensure_ascii=False)
