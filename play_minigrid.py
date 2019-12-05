from __future__ import division, print_function

import sys
import torch
import gym
import time
import json
import os
from optparse import OptionParser
from datetime import datetime

import gym_minigrid

games_path = './games/'

game_name = None
game_info = {
    'name': game_name,
    'trajectory': [],
    'score': None
}
directory = None


def state_filter(state):
    """
    :param state: current state (with 3 channels)
    :return: only image state tensor (1^ channel)
    """
    return torch.from_numpy(state['image'][:, :, 0]).float()


def reset_env(env):
    """
    reset the environment, initialize game_name, game_info and directory

    :param env: gym environment used
    :return:
    """
    global game_name, game_info, directory
    env.reset()
    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)

    # Get timestamp to identify this game
    game_name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    print('New game: ', game_name)

    # Dictionary for game information
    game_info = {
        'name': game_name,
        'trajectory': [],
        'score': None
    }

    # Create new folder to save images
    directory = games_path + str(game_name)
    if not os.path.exists(directory):
        os.makedirs(directory)


def act_action(env, action):
    """
    calculate new state (obs), save image of the state and if finished reset the environment

    :param env: gym environment used
    :param action: action taken
    :return:
    """
    obs, reward, done, info = env.step(action)
    print("state: ", state_filter(obs))

    # Save state
    game_info['trajectory'].append(state_filter(obs).tolist())

    print('step=%s, reward=%.2f' % (env.step_count, reward))

    # Save image of each state
    screen_path = games_path + str(game_name) + '/game' + str(env.step_count) + '.png'
    pixmap = env.render('pixmap')
    pixmap.save(screen_path)

    if done:
        with open(games_path + str(game_name) + '.json', 'w+') as game_file:
            json.dump(game_info, game_file, ensure_ascii=False)

        print('done!', len(game_info['trajectory']))
        reset_env(env)


def main():

    global game_name, game_info
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-Empty-6x6-v0'
    )
    (options, args) = parser.parse_args()

    # Load the gym environment
    env = gym.make(options.env_name)

    reset_env(env)

    # Create a window to render into
    renderer = env.render('human')

    def keyDownCb(keyName):
        if keyName == 'BACKSPACE':
            reset_env(env)
            return

        if keyName == 'ESCAPE':
            sys.exit(0)

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

    renderer.window.setKeyDownCb(keyDownCb)

    while True:
        env.render('human')
        time.sleep(0.01)

        # If the window was closed
        if renderer.window is None:
            break


if __name__ == "__main__":
    main()

