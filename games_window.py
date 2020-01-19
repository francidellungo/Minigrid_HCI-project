from datetime import datetime
import sys
import gym
import os
import json

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QEvent
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QLabel, QPushButton, QHBoxLayout
from games_model import GamesModel
from games_view import GamesView
from Ui_scrollbar_v2 import Ui_MainWindow
from Ui_newGame import Ui_new_game_Dialog
# from play_minigrid import main
# from play_minigrid_one import *

env_used = 'MiniGrid-Empty-6x6-v0'
games_path = 'games'

folder_name = '2019-12-11_13:40:14'


class GamesController:

    def __init__(self, env, agents_window, agents_model=None):

        # env
        self.env = env

        # _agents_model
        self.agents_model = agents_model

        # games_model
        self.games_model = GamesModel(env, agents_model)

        # view (gui)
        self.view = GamesView(env, self.games_model, agents_window, self.agents_model)
        self.view.show()

        # gui for new game
        # self.new_game_view_Dialog = NewGameView(env)

        # connect games_model signals to slots
        self.games_model.new_game_s.connect(self.view.add_row)
        # self.games_model.new_game_s.connect(self.create_delete_pb_connection)

        self.games_model.game_removed.connect(self.view.remove_game_from_gui)
        self.games_model.game_moved.connect(self.view.move_game_gui)

        self.games_model.moved_up.connect(self.view.move_game_down_gui)
        self.games_model.moved_down.connect(self.view.move_game_up_gui)

        # connect buttons events to slots (without view)
        # self.ui.new_game_pb.clicked.connect(lambda: self.games_model.new_game(env_used, 'game ' + str(self.games_model.n_games)))

        # connect buttons events to slots
        self.view.ui.new_game_pb.clicked.connect(lambda: self.create_new_game(env))

    def create_new_game(self, env):
        """
        creation of a new game
        :param env: current environment used
        :param name: name for the new game (really needed?)
        :return:
        """
        self.new_game_dialog = NewGameView(env, self.games_model)
        self.new_game_dialog.exec_()


class NewGameView(QDialog):

    game_saved = pyqtSignal(str, str, str)  # str: game_folder name

    def __init__(self, environment, games_model):
        super().__init__()
        self.ui = Ui_new_game_Dialog()
        self.ui.setupUi(self)
        self.games_model = games_model
        self.env = gym.make(environment)
        self.env_name = environment
        # self.keyDownCb = None
        self.done = False
        self.game_folder = None
        self.ui.game_buttonBox.button(QtWidgets.QDialogButtonBox.Save).setEnabled(False)

        self.reset_env(self.env_name)
        pixmap = self.env.render('pixmap')
        self.ui.game_label.setPixmap(pixmap)

        self.show()

    def accept(self) -> str:
        global game_directory

        k = 1
        print('accept event _ ', self.game_folder)
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

        # self.game_saved.emit(self.env_name, self.game_folder, 'game_ ')
        self.games_model.new_game(self.env_name, self.game_folder, self.game_folder)
        self.close()
        return self.game_folder

    def act_action(self, action):
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

        obs, reward, done, info = self.env.step(action)
        pixmap = self.env.render('pixmap')
        self.ui.game_label.setPixmap(pixmap)

        print("state: ", self.state_filter(obs))

        # Save state
        game_info['trajectory'].append(self.state_filter(obs).tolist())
        game_info['rewards'].append(reward)

        print('step=%s, reward=%.2f' % (self.env.step_count, reward))

        # Save screenshots
        screenshot_file = 'game' + str(self.env.step_count) + '.png'
        pixmap = self.env.render('pixmap')
        screenshots.append((screenshot_file, pixmap))

        if done:
            print('done!', len(game_info['trajectory']))

            pixmap = self.env.render('pixmap')
            self.ui.game_label.setPixmap(pixmap)
            self.env.close()
            self.ui.game_buttonBox.button(QtWidgets.QDialogButtonBox.Save).setEnabled(True)
            self.done = True

        # if action == self.env.actions.done:
        #     return obs, None, True, None
        return obs, reward, done, info

    def state_filter(self, state):
        return state['image'][:, :, 0]

    def reset_env(self, env_name):
        """
        reset the environment, initialize game_name, game_info and directory

        :param env: gym environment used
        :return:
        """
        global game_name, game_info, game_directory, screenshots
        state = self.env.reset()

        # if hasattr(env, 'mission'):
        #     print('Mission: %s' % self.env.mission)

        # Get timestamp to identify this game
        game_name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        print('New game: ', game_name)

        # Dictionary for game information
        game_info = {
            'name': game_name,
            'trajectory': [self.state_filter(state).tolist()],
            'rewards': [0],
            'score': None,
            'to_delete': False
        }

        game_directory = os.path.join(games_path, env_name, str(game_name))

        screenshot_file = 'game' + str(self.env.step_count) + '.png'
        pixmap = self.env.render('pixmap')
        screenshots = [(screenshot_file, pixmap)]

        self.game_folder = game_name

        return state, game_directory, pixmap

    def keyPressEvent(self, QKeyEvent):
        # print('key press event', QEvent.KeyPress)
        if QKeyEvent.type() == QEvent.KeyPress and not self.done:
            if QKeyEvent.key() == Qt.Key_A:
                action = self.env.actions.left
                # self.sig_key_left.emit()
                # QKeyEvent.accept()

            elif QKeyEvent.key() == Qt.Key_D:
                action = self.env.actions.right
                # self.sig_key_right.emit()
                # QKeyEvent.accept()

            elif QKeyEvent.key() == Qt.Key_W:
                action = self.env.actions.forward
                # self.sig_key_home.emit()
                # QKeyEvent.accept()
            elif QKeyEvent.key() == Qt.Key_P:
                action = self.env.actions.pickup
            elif QKeyEvent.key() == Qt.Key_O:
                action = self.env.actions.drop
            elif QKeyEvent.key() == Qt.Key_I:
                action = self.env.actions.toggle

            else:
                print("unknown key %s" % QKeyEvent.key())
                return

            # elif QKeyEvent.key() == Qt.Key_End:
            #     self.sig_key_end.emit()
            #     QKeyEvent.accept()

            self.act_action(action)

#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = GamesController(env_used)
#     # window.show()
#     sys.exit(app.exec_())
