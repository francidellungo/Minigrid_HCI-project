# from __future__ import division, print_function
import sys
import torch
import gym
import time
import json
import os
from optparse import OptionParser
from datetime import datetime
from glob import glob

import gym_minigrid

games_path = 'games'

game_name = None
game_info = {
    'name': game_name,
    'trajectory': [],
    'rewards': None,
    'score': None,
    'to_delete': False
}
# to delete == True if the trajectory is deleted from the player (useful for the graphical interface)
game_directory = None
screenshots = []


def state_filter(state):
    """
    :param state: current state (with 3 channels)
    :return: only image state tensor (1^ channel)
    """
    return torch.from_numpy(state['image'][:, :, 0]).float().to(device)


def reset_env(env):
    """
    reset the environment, initialize game_name, game_info and directory

    :param env: gym environment used
    :return:
    """
    global game_name, game_info, game_directory, screenshots
    state = env.reset()
    env.render()
    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)

    # Get timestamp to identify this game
    game_name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    print('New game: ', game_name)

    # Dictionary for game information
    game_info = {
        'name': game_name,
        'trajectory': [state_filter(state).tolist()],
        'rewards': [0],
        'score': None,
        'to_delete': False
    }

    game_directory = os.path.join(games_path, options.env_name, str(game_name))

    screenshot_file = 'game' + str(env.step_count) + '.png'
    pixmap = env.render('pixmap')
    screenshots = [(screenshot_file, pixmap)]

    return state


def act_action(env, action):
    """
    calculate new state (obs), save image of the state and if finished reset the environment

    :param env: gym environment used
    :param action: action taken
    :return:
    """
    global game_directory
    if action == env.actions.done:
        done = True
    else:
        obs, reward, done, info = env.step(action)
        print("state: ", state_filter(obs))

        # Save state
        game_info['trajectory'].append(state_filter(obs).tolist())
        game_info['rewards'].append(reward)

        print('step=%s, reward=%.2f' % (env.step_count, reward))

        # Save screenshots
        screenshot_file = 'game' + str(env.step_count) + '.png'
        pixmap = env.render('pixmap')
        screenshots.append((screenshot_file, pixmap))

    if done:
        # Save images and json

        # Create new folder to save images and json
        k = 1
        original_game_directory = game_directory
        while os.path.exists(game_directory):
            game_directory = original_game_directory + "_" + str(k)
            k += 1
        os.makedirs(game_directory)

        # Save image of each state
        for screenshot_file, pixmap in screenshots:
            pixmap.save(os.path.join(game_directory, screenshot_file))

        game_info["score"] = sum(game_info["rewards"])
        with open(os.path.join(game_directory, 'game.json'), 'w+') as game_file:
            json.dump(game_info, game_file, ensure_ascii=False)

        print('done!', len(game_info['trajectory']))
        obs = reset_env(env)

    if action == env.actions.done:
        return obs, None, True, None
    return obs, reward, done, info


def keyDownCb(keyName):

        if keyName == 'ESCAPE':
            sys.exit(0)

        if options.policy_net is not None:
            # controls are disabled if human user isn't playing
            return

        if keyName == 'BACKSPACE':
            reset_env(env)
            return

        action = 0

        if keyName == 'LEFT':
            action = env.actions.left
        elif keyName == 'RIGHT':
            action = env.actions.right
        elif keyName == 'UP':
            action = env.actions.forward

        elif keyName == 'SPACE':
            action = env.actions.toggle
        elif keyName == 'PAGE_UP':
            action = env.actions.pickup
        elif keyName == 'PAGE_DOWN':
            action = env.actions.drop

        elif keyName == 'RETURN':
            action = env.actions.done
            #action = 'exit_game'

        # Screenshot functionality
        elif keyName == 'ALT':
            screen_path = options.env_name + '.png'
            print('saving screenshot "{}"'.format(screen_path))
            pixmap = env.render('pixmap')
            pixmap.save(screen_path)
            return

        else:
            print("unknown key %s" % keyName)
            return

        # Update state
        act_action(env, action)


def main():

    global game_name, game_info, options, env, device
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-Empty-6x6-v0'
    )
    parser.add_option(
        "-n",
        "--num-episodes",
        dest="num_episodes",
        help="max number of episodes to play (only valid for non-human player)",
        default=-1
    )
    parser.add_option(
        "-p",
        "--policy",
        dest="policy_net",
        help="policy net to use as agent. Default: no policy, the user is the agent",
        default=None
    )
    (options, args) = parser.parse_args()

    # use GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the gym environment
    env = gym.make(options.env_name)

    state = reset_env(env)

    # Create a window to render into
    renderer = env.render('human')

    # set controls
    renderer.window.setKeyDownCb(keyDownCb)

    if options.policy_net is not None:

        if options.policy_net.endswith(".pth"):
            # select specified weights
            epoch_to_load_weights = options.policy_net.rsplit("-", 1)[1].split(".", 1)[0]
            policy_net_dir = os.path.dirname(options.policy_net)
        else:
            # load the most recent weights from the specified folder
            episodes_saved_weights = [int(state.rsplit("-", 1)[1].split(".", 1)[0]) for state in glob(os.path.join(options.policy_net, "policy_net-*.pth"))]
            epoch_to_load_weights = max(episodes_saved_weights)
            policy_net_dir = options.policy_net

        agent = torch.load(os.path.join(policy_net_dir, "net.pth"))
        agent.load_state_dict(torch.load(os.path.join(policy_net_dir, "policy_net-" + str(epoch_to_load_weights) + ".pth")))
        agent = agent.to(device)

    done = False
    count = 0
    while True:
        env.render('human')

        if done:
            count += 1
            if options.policy_net is not None and count >= int(options.num_episodes):
                break

        if options.policy_net is not None:
            action = agent.sample_action(state_filter(state))
            state, r, done, _ = act_action(env, action)

        # If the window was closed
        if renderer.window is None:
            break


def minigrid_play_one():

    global game_name, game_info, options, env, device
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-Empty-6x6-v0'
    )
    parser.add_option(
        "-n",
        "--num-episodes",
        dest="num_episodes",
        help="max number of episodes to play (only valid for non-human player)",
        default=-1
    )
    parser.add_option(
        "-p",
        "--policy",
        dest="policy_net",
        help="policy net to use as agent. Default: no policy, the user is the agent",
        default=None
    )
    (options, args) = parser.parse_args()

    # use GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the gym environment
    env = gym.make(options.env_name)

    state = reset_env(env)

    # Create a window to render into
    renderer = env.render('human')

    # set controls
    renderer.window.setKeyDownCb(keyDownCb)

    if options.policy_net is not None:

        if options.policy_net.endswith(".pth"):
            # select specified weights
            epoch_to_load_weights = options.policy_net.rsplit("-", 1)[1].split(".", 1)[0]
            policy_net_dir = os.path.dirname(options.policy_net)
        else:
            # load the most recent weights from the specified folder
            episodes_saved_weights = [int(state.rsplit("-", 1)[1].split(".", 1)[0]) for state in glob(os.path.join(options.policy_net, "policy_net-*.pth"))]
            epoch_to_load_weights = max(episodes_saved_weights)
            policy_net_dir = options.policy_net

        agent = torch.load(os.path.join(policy_net_dir, "net.pth"))
        agent.load_state_dict(torch.load(os.path.join(policy_net_dir, "policy_net-" + str(epoch_to_load_weights) + ".pth")))
        agent = agent.to(device)

    done = False
    count = 0
    while not done:

        env.render('human')
        # if done:
        #     count += 1
        #     if options.policy_net is not None and count >= int(options.num_episodes):
        #         break
        #
        # if options.policy_net is not None:
        #     action = agent.sample_action(state_filter(state))
        #     state, r, done, _ = act_action(env, action)

        # If the window was closed
        if renderer.window is None:
            break


if __name__ == "__main__":
    minigrid_play_one()

