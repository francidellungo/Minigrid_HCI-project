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
    # if action == env.actions.done:
    #     done = True
    # else:
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

        sys.exit(0)
        # TODO change

    if action == env.actions.done:
        return obs, None, True, None
    return obs, reward, done, info


def keyDownCb(keyName, env):

    if keyName == 'ESCAPE':
        sys.exit(0)

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

    # elif keyName == 'RETURN':
    #     action = env.actions.done
    #     #action = 'exit_game'

    # Screenshot functionality
    # elif keyName == 'ALT':
    #     screen_path = options.env_name + '.png'
    #     print('saving screenshot "{}"'.format(screen_path))
    #     pixmap = env.render('pixmap')
    #     pixmap.save(screen_path)
    #     return

    else:
        print("unknown key %s" % keyName)
        return

    # Update state
    # act_action(env, action)
    act_action(env, action)


def minigrid_play_one(env_used):

    global game_name, game_info, options, env, device

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

    # Load the gym environment
    env = gym.make(options.env_name)

    state = reset_env(env)

    # Create a window to render into
    renderer = env.render('human')

    # set controls
    renderer.window.setKeyDownCb(keyDownCb)

    done = False
    while not done:
        env.render('human')

        if renderer.window is None:
            break


if __name__ == "__main__":
    minigrid_play_one()

