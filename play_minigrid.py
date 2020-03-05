import json
import sys
import time
from random import randint
from threading import Thread

import torch
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout
from termcolor import colored

import os
import argparse
from datetime import datetime
import collections

import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window

from policy_nets.base_policy_net import PolicyNet
from utils import *


class Game:
    def __init__(self, env, seed=-1, agent_view=False, games_directory=None, refresh_callback=None, end_callback=None, handle_special_keys=True, policy_net=None, max_games=-1, waiting_time=0, reward_net=None, autosave=True):
        self.env = env
        self.seed = seed
        self.agent_view = agent_view
        self.games_directory = games_directory
        self.handle_special_keys = handle_special_keys
        self.policy_net = policy_net
        self.max_games = max_games
        self.waiting_time = waiting_time
        self.reward_net = reward_net
        self.autosave = autosave

        self.num_games_ended = 0

        self.env_width = env.width - 2

        # create target sequence of action (desired trajectory for the agent)
        seq = [2 for i in range(self.env_width-1)]

        self.target_traj = [1] + seq + [0] + seq
        self.target_traj_tt = [0, 0, 0] + seq + [0] + seq

        # self.target_traj = [1, 2, 2, 2, 0, 2, 2, 2]
        # self.target_traj_tt = [0, 0, 0, 2, 2, 2, 0, 2, 2, 2]
        self.curr_traj = []
        self.count_traj = 0

        self.traj_dict = {}

        if games_directory is not None:
            self.game_name = None
            self.game_info = {
                'name': self.game_name,
                'trajectory': [],
                'rewards': None,
                'score': None,
                'to_delete': False
            }
            # to delete == True if the trajectory is deleted from the game (useful for the graphical interface)
            self.screenshots = []
            self.folder = None

        if refresh_callback is not None:
            self.refresh_gui = refresh_callback
        self.end_callback = end_callback

        self.reset()

        if policy_net is not None:
            self.thread = Thread(target=self._autoplay)
            self.thread.start()

        mem = 0.99
        self.env_standardizer = Standardizer(mem)
        self.net_standardizer = Standardizer(mem)
        self.env_disc_standardizer = Standardizer(mem)
        self.net_disc_standardizer = Standardizer(mem)

        history_length = 500
        self.env_sum_standardizer = SumStandardizer(history_length)
        self.net_sum_standardizer = SumStandardizer(history_length)
        self.env_disc_sum_standardizer = SumStandardizer(history_length)
        self.net_disc_sum_standardizer = SumStandardizer(history_length)

    def refresh_gui(self, np_array):
        pass

    def _refresh_gui(self, obs):
        if self.agent_view:
            self.refresh_gui(obs)
        else:
            img = self.env.render('pixmap')
            self.refresh_gui(img)

    def reset(self):
        """
        reset the environment, initialize game_name, game_info and directory

        :param env: gym environment used
        :return:
        """
        if self.num_games_ended == self.max_games:
            return

        if self.seed != -1:
            self.env.seed(self.seed)
            print("seed {} set".format(self.seed))
        else:
            self.env.seed(randint(0, 1000000))

        self.obs = self.env.reset()
        self.tot_env_reward = 0
        self.tot_net_reward = 0
        self.env_rewards = []
        self.net_rewards = []

        if hasattr(self.env, 'mission'):
            print('Mission: %s' % self.env.mission)

        self._refresh_gui(self.obs)

        if self.games_directory is not None:
            # Get timestamp to identify this game
            self.game_name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            print('New game: ', self.game_name)

            # Dictionary for game information
            self.game_info = {
                'name': self.game_name,
                'trajectory': [state_filter(self.obs).tolist()],
                'rewards': [0],
                'score': None,
                'to_delete': False
            }

            self.folder = os.path.join(self.games_directory, self.env.unwrapped.spec.id, str(self.game_name))

            screenshot_file = 'game' + str(self.env.step_count) + '.png'
            pixmap = self.env.render('pixmap')
            self.screenshots = [(screenshot_file, pixmap)]

        return self

    def step(self, action):
        # print('action: ', int(action))
        self.curr_traj.append(int(action))

        if self.num_games_ended == self.max_games:
            return
        obs, reward, done, info = self.env.step(action)
        self.obs = self.env.gen_obs()
        print_observation(self.obs)
        self.print_step_details(reward, self.obs)

        if self.games_directory is not None:
            # Save state
            self.game_info['trajectory'].append(state_filter(self.obs).tolist())
            self.game_info['rewards'].append(reward)
            # print('rewards', self.game_info['rewards'])
            # Save screenshots
            screenshot_file = 'game' + str(self.env.step_count) + '.png'
            pixmap = self.env.render('pixmap')
            self.screenshots.append((screenshot_file, pixmap))

        self._refresh_gui(self.obs)
        if done:
            print('done!')

            # try use normalized reward
            # print(self.game_info['rewards'])
            # self.game_info['rewards'] = normalize(self.game_info['rewards'])
            # print('normalized', self.game_info['rewards'])

            self.print_rewards()
            self.print_discounted_rewards()
            if self.games_directory is not None and self.autosave:
                self.save()

            self.num_games_ended += 1
            if self.num_games_ended == self.max_games:
                self._notify_end()
                self.interrupt()
            else:
                self.reset()

            # get statistic info
            if len(self.curr_traj) in self.traj_dict:
                self.traj_dict[len(self.curr_traj)] += 1
            else:
                self.traj_dict[len(self.curr_traj)] = 1

            # print('curr_traj', self.curr_traj, len(self.curr_traj))
            if self.curr_traj == self.target_traj:
                self.count_traj += 1
            self.curr_traj = []

            print('correct trajectories: ', self.count_traj, ' / ', self.max_games, ' = ', self.count_traj/self.max_games * 100, '%')
            print(self.traj_dict)
            print(sorted(self.traj_dict.items()))

        return self

    def _notify_end(self):
        if self.end_callback is not None:
            self.end_callback()

    def plt_key_handler(self, event):
        if self.handle_special_keys and event.key == 'escape':
                exit(0)
        if self.num_games_ended == self.max_games:
            return
        print('\npressed', event.key)
        if event.key == 'left':
            action = self.env.actions.left
        elif event.key == 'right':
            action = self.env.actions.right
        elif event.key == 'up':
            action = self.env.actions.forward
        elif event.key == ' ':  # Spacebar
            action = self.env.actions.toggle
        elif event.key == 'pageup':
            action = self.env.actions.pickup
        elif event.key == 'pagedown':
            action = self.env.actions.drop
        elif self.handle_special_keys and event.key == 'enter':
            action = self.env.actions.done
        elif self.handle_special_keys and event.key == 'backspace':
            self.reset()
            return
        else:
            print("\nunknown key %s" % event.key)
            return
        self.step(action)

    def qt_key_handler(self, qt_key_event):
        if qt_key_event.type() != qt_key_event.KeyPress:
            return
        if self.handle_special_keys and qt_key_event.key() == Qt.Key_Escape:
            exit(0)
        if self.num_games_ended == self.max_games:
            return
        print("\npressed " + qt_key_event.text())
        if qt_key_event.key() == Qt.Key_A or qt_key_event.key() == Qt.Key_Left:
            action = self.env.actions.left
        elif qt_key_event.key() == Qt.Key_D or qt_key_event.key() == Qt.Key_Right:
            action = self.env.actions.right
        elif qt_key_event.key() == Qt.Key_W or qt_key_event.key() == Qt.Key_Up:
            action = self.env.actions.forward
        elif qt_key_event.key() == Qt.Key_P or qt_key_event.key() == Qt.Key_PageUp:
            action = self.env.actions.pickup
        elif qt_key_event.key() == Qt.Key_O or qt_key_event.key() == Qt.Key_PageDown:
            action = self.env.actions.drop
        elif qt_key_event.key() == Qt.Key_I or qt_key_event.key() == Qt.Key_Space:
            action = self.env.actions.toggle
        elif self.handle_special_keys and qt_key_event.key() == Qt.Key_Enter:
            action = self.env.actions.done
        elif self.handle_special_keys and qt_key_event.key() == Qt.Key_Backspace:
            self.reset()
            return
        else:
            print("\nunknown key %s" % qt_key_event.key())
            return

        self.step(action)

    def save(self):
        """
        Save images and json
        :return: None
        """
        # Create new folder to save images and json
        k = 1
        original_folder = self.folder
        while os.path.exists(self.folder):
            self.folder = original_folder + "_" + str(k)
            k += 1
        os.makedirs(self.folder)

        # Save image of each state
        for screenshot_file, img in self.screenshots:
            pixmap = QPixmap(QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888))
            pixmap.save(os.path.join(self.folder, screenshot_file))

        self.game_info["score"] = sum(self.game_info["rewards"])
        with open(os.path.join(self.folder, 'game.json'), 'w+') as game_file:
            json.dump(self.game_info, game_file, ensure_ascii=False)

    def _autoplay(self):
        self._running = True
        while self._running:
            action = self.policy_net.sample_action(state_filter(self.obs))
            self.step(action)
            if self.waiting_time > 0:
                time.sleep(self.waiting_time)

    def interrupt(self):
        self._running = False

    def print_step_details(self, env_reward, obs):
        self.env_rewards.append(env_reward)
        self.tot_env_reward += env_reward
        output = 'step=%s\nenv_reward=%.2f, tot_env_reward=%.2f' % (self.env.step_count, env_reward, self.tot_env_reward)
        if self.reward_net is not None:
            net_reward = self.reward_net(state_filter(obs), torch.tensor([self.env.step_count])).item()
            self.net_rewards.append(net_reward)
            self.tot_net_reward += net_reward
            output += '\nnet_reward=%.2f, tot_net_reward=%.2f' % (net_reward, self.tot_net_reward)
        print(output)

    def print_rewards(self):
        output = "env_rewards: " + str(rounded_list(self.env_rewards))
        if self.reward_net is not None:
            output += "\nnet_rewards: " + str(rounded_list(self.net_rewards))
        output += "\nenv_normalized_rewards: " + str(rounded_list(normalize(self.env_rewards)))
        if self.reward_net is not None:
            output += "\nnet_normalized_rewards: " + str(rounded_list(normalize(self.net_rewards)))
        output += "\nenv_standardized_rewards: " + str(rounded_list(standardize(self.env_rewards)))
        if self.reward_net is not None:
            output += "\nnet_standardized_rewards: " + str(rounded_list(standardize(self.net_rewards)))
        output += "\nenv_standardized_rewards_with_memory: " + str(rounded_list(self.env_standardizer.standardize(self.env_rewards)))
        if self.reward_net is not None:
            output += "\nnet_standardized_rewards_with_memory: " + str(rounded_list(self.net_standardizer.standardize(self.net_rewards)))
        output += "\nenv_standardized_rewards_sum: " + str(rounded_list(self.env_sum_standardizer.standardize(self.env_rewards)))
        if self.reward_net is not None:
            output += "\nnet_standardized_rewards_sum: " + str(rounded_list(self.net_sum_standardizer.standardize(self.net_rewards)))
        print(output)

    def print_discounted_rewards(self):
        env_discounted_rewards = PolicyNet.compute_discounted_rewards(self.env_rewards)
        output = "env_discounted_rewards: " + str(rounded_list(env_discounted_rewards))
        if self.reward_net is not None:
            net_discounted_rewards = PolicyNet.compute_discounted_rewards(self.net_rewards)
            output += "\nnet_discounted_rewards: " + str(rounded_list(net_discounted_rewards))
        output += "\nenv_normalized_discounted_rewards: " + str(rounded_list(normalize(env_discounted_rewards)))
        if self.reward_net is not None:
            output += "\nnet_normalized_discounted_rewards: " + str(rounded_list(normalize(net_discounted_rewards)))
        output += "\nenv_standardized_discounted_rewards: " + str(rounded_list(standardize(env_discounted_rewards)))
        if self.reward_net is not None:
            output += "\nnet_standardized_discounted_rewards: " + str(rounded_list(standardize(net_discounted_rewards)))
        output += "\nenv_standardized_discounted_rewards_with_memory: " + str(rounded_list(self.env_disc_standardizer.standardize(env_discounted_rewards)))
        if self.reward_net is not None:
            output += "\nnet_standardized_discounted_rewards_with_memory: " + str(rounded_list(self.net_disc_standardizer.standardize(net_discounted_rewards)))
        output += "\nenv_standardized_discounted_rewards_sum: " + str(rounded_list(self.env_disc_sum_standardizer.standardize(env_discounted_rewards)))
        if self.reward_net is not None:
            output += "\nnet_standardized_discounted_rewards_sum: " + str(rounded_list(self.net_disc_sum_standardizer.standardize(net_discounted_rewards)))
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--backend", help="Backend to use. Default: qt", default='qt', choices=['qt', 'plt'])
    parser.add_argument("-e", "--env", help="Gym environment to load. Default: MiniGrid-Empty-6x6-v0", default='MiniGrid-Empty-6x6-v0', choices=get_all_environments())
    parser.add_argument("-s", "--seed", type=int, help="Random seed to generate the environment with", default=-1)
    parser.add_argument("-av", '--agent_view', default=False, help="Draw the agent sees (partially observable view). Default: False", action='store_true')
    parser.add_argument("-g", "--games_dir", help="Directory where to save games. Default: games aren't saved", default=None)
    parser.add_argument("-p", "--policy_net", help="Policy net to use as agent. Default: no policy_net, the game is the user", default=None)
    parser.add_argument("-r", "--reward_net", help="Reward net to evalute. Default: None", default=None)
    parser.add_argument("-mg", "--max_games", help="Maximum number of games to play. Default: no limits", type=int, default=-1)
    parser.add_argument("-wt", "--waiting_time", help="Policy waiting time (seconds) between moves. Default: 0", type=float, default=0)

    args = parser.parse_args()

    env = gym.make(args.env)

    if args.agent_view:
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)

    policy_net = load_net(args.policy_net, True)
    reward_net = load_net(args.reward_net, True)

    if args.backend == "qt":
        app = QApplication(sys.argv)
        window = QMainWindow()
        central_widget = QWidget()
        v_layout = QVBoxLayout(central_widget)
        widget_game = QLabel("")
        widget_caption = QLabel("")
        v_layout.addWidget(widget_game)
        v_layout.addWidget(widget_caption)
        window.setCentralWidget(central_widget)
        redraw = lambda img: (widget_game.setPixmap(nparray_to_qpixmap(img)), widget_caption.setText(env.mission))
        game = Game(env, args.seed, args.agent_view, args.games_dir, redraw, lambda:..., True, policy_net, args.max_games, args.waiting_time, reward_net)
        window.keyPressEvent = game.qt_key_handler
        window.show()
        sys.exit(app.exec_())
    elif args.backend == "plt":
        window = Window('gym_minigrid - ' + args.env)
        redraw = lambda img: (window.show_img(img), window.set_caption(env.mission))
        game = Game(env, args.seed, args.agent_view, args.games_dir, redraw, lambda:..., True, policy_net, args.max_games, args.waiting_time, reward_net)
        window.reg_key_handler(game.plt_key_handler)
        # Blocking event loop
        window.show(block=True)
    else:
        print("unknown backend")
