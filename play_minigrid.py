# from __future__ import division, print_function
import sys
import torch
import gym
import time
import json
import os
from optparse import OptionParser
from datetime import datetime

import gym_minigrid

games_path = 'games'

game_name = None
game_info = {
    'name': game_name,
    'trajectory': [],
    'score': None
}
game_directory = None
screenshots = []


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
    global game_name, game_info, game_directory, screenshots
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

    game_directory = os.path.join(games_path, options.env_name, str(game_name))
    screenshots = []


def act_action(env, action):
    """
    calculate new state (obs), save image of the state and if finished reset the environment

    :param env: gym environment used
    :param action: action taken
    :return:
    """

    if action == 'exit_game':
        done = True
    else:
        obs, reward, done, info = env.step(action)
        print("state: ", state_filter(obs))

        # Save state
        game_info['trajectory'].append(state_filter(obs).tolist())

        print('step=%s, reward=%.2f' % (env.step_count, reward))

        # Save screenshots
        screenshot_path = os.path.join(game_directory, 'game' + str(env.step_count) + '.png')
        pixmap = env.render('pixmap')
        screenshots.append((screenshot_path, pixmap))

    if done:
        # Save images and json

        # Create new folder to save images and json
        if not os.path.exists(game_directory):
            os.makedirs(game_directory)

        # Save image of each state
        for screenshot_path, pixmap in screenshots:
            pixmap.save(screenshot_path)

        with open(os.path.join(game_directory, 'game.json'), 'w+') as game_file:
            json.dump(game_info, game_file, ensure_ascii=False)

        print('done!', len(game_info['trajectory']))
        reset_env(env)


def main():

    global game_name, game_info, options
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
            # action = env.actions.done
            action = 'exit_game'

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

