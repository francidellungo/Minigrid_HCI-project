import json
import sys
import time
from threading import Thread

import torch
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout
from termcolor import colored

import os
import argparse
from datetime import datetime

import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window

from utils import state_filter, nparray_to_qpixmap, print_observation, get_all_environments, load_policy


class Game:
    def __init__(self, env, seed=-1, agent_view=False, games_directory=None, refresh_callback=None, end_callback=None, handle_special_keys=True, policy=None, max_games=-1, waiting_time=0, autosave=True):
        self.env = env
        self.seed = seed
        self.agent_view = agent_view
        self.games_directory = games_directory
        self.handle_special_keys = handle_special_keys
        self.policy = policy
        self.max_games = max_games
        self.waiting_time = waiting_time
        self.autosave = autosave
        self.num_games_ended = 0

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

        if policy is not None:
            self.thread = Thread(target=self._autoplay)
            self.thread.start()

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

        self.obs = self.env.reset()

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
        if self.num_games_ended == self.max_games:
            return
        obs, reward, done, info = self.env.step(action)
        self.obs = obs
        print_observation(self.obs)
        print('step=%s, reward=%.2f' % (self.env.step_count, reward))

        if self.games_directory is not None:
            # Save state
            self.game_info['trajectory'].append(state_filter(obs).tolist())
            self.game_info['rewards'].append(reward)
            # Save screenshots
            screenshot_file = 'game' + str(self.env.step_count) + '.png'
            pixmap = self.env.render('pixmap')
            self.screenshots.append((screenshot_file, pixmap))

        self._refresh_gui(obs)
        if done:
            print('done!')
            if self.games_directory is not None and self.autosave:
                self.save()

            self.num_games_ended += 1
            if self.num_games_ended == self.max_games:
                self._notify_end()
                self.interrupt()
            else:
                self.reset()

        return self

    def _notify_end(self):
        if self.end_callback is not None:
            self.end_callback()

    def plt_key_handler(self, event):
        if self.handle_special_keys and event.key == 'escape':
                exit(0)
        if self.num_games_ended == self.max_games:
            return
        print('pressed', event.key)
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
            print("unknown key %s" % event.key)
            return
        self.step(action)

    def qt_key_handler(self, qt_key_event):
        if qt_key_event.type() != qt_key_event.KeyPress:
            return
        if self.handle_special_keys and qt_key_event.key() == Qt.Key_Escape:
            exit(0)
        if self.num_games_ended == self.max_games:
            return
        print("pressed " + qt_key_event.text())
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
            print("unknown key %s" % qt_key_event.key())
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
            action = self.policy.sample_action(state_filter(self.obs))
            self.step(action)
            if self.waiting_time > 0:
                time.sleep(self.waiting_time)

    def interrupt(self):
        self._running = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--backend", help="Backend to use. Default: qt", default='qt', choices=['qt', 'plt'])
    parser.add_argument("-e", "--env", help="Gym environment to load. Default: MiniGrid-Empty-6x6-v0", default='MiniGrid-Empty-6x6-v0', choices=get_all_environments())
    parser.add_argument("-s", "--seed", type=int, help="Random seed to generate the environment with", default=-1)
    parser.add_argument("-av", '--agent_view', default=False, help="Draw the agent sees (partially observable view). Default: False", action='store_true')
    parser.add_argument("-g", "--games_dir", help="Directory where to save games. Default: games aren't saved", default=None)
    parser.add_argument("-p", "--policy", help="Policy net to use as agent. Default: no policy, the game is the user", default=None)
    parser.add_argument("-r", "--reward", help="Reward net to evalute. Default: None", default=None)
    parser.add_argument("-mg", "--max_games", help="Maximum number of games to play. Default: no limits", type=int, default=-1)
    parser.add_argument("-wt", "--waiting_time", help="Policy waiting time (seconds) between moves. Default: 0", type=float, default=0)

    args = parser.parse_args()

    env = gym.make(args.env)

    if args.agent_view:
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)

    policy = load_policy(args.policy)

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
        game = Game(env, args.seed, args.agent_view, args.games_dir, redraw, True, policy, args.max_games, args.waiting_time)
        window.keyPressEvent = game.qt_key_handler
        window.show()
        sys.exit(app.exec_())
    elif args.backend == "plt":
        window = Window('gym_minigrid - ' + args.env)
        redraw = lambda img: (window.show_img(img), window.set_caption(env.mission))
        game = Game(env, args.seed, args.agent_view, args.games_dir, redraw, True, policy, args.max_games, args.waiting_time)
        window.reg_key_handler(game.plt_key_handler)
        # Blocking event loop
        window.show(block=True)
    else:
        print("unknown backend")
