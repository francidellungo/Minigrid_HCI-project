import json
from datetime import datetime
import sys
import gym
import os
import json

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QEvent
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QLabel, QPushButton, QHBoxLayout
from sklearn.ensemble._gradient_boosting import np_float32

from games_model import GamesModel
from games_view import GamesView
from Ui_scrollbar_v2 import Ui_MainWindow
from Ui_newGame import Ui_new_game_Dialog
from play_minigrid import Game
from utils import *

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

    def __init__(self, environment, games_model):
        super().__init__()
        self.ui = Ui_new_game_Dialog()
        self.ui.setupUi(self)
        self.games_model = games_model
        self.env = gym.make(environment)
        self.env_name = environment
        self.ui.game_buttonBox.button(QtWidgets.QDialogButtonBox.Save).setEnabled(False)
        # self.ui.game_buttonBox.accepted.connect(self.accept)

        self.show()
        self.game = Game(self.env, games_directory=games_dir(), refresh_callback=self.update_gui, end_callback=self.enable_save, handle_special_keys=False, max_games=1, autosave=False)
        self.ui.game_buttonBox.button(QtWidgets.QDialogButtonBox.Save).clicked.connect(self.on_save)
        self.keyPressEvent = self.game.qt_key_handler

    def update_gui(self, img):
        self.ui.game_label.setPixmap(nparray_to_qpixmap(img))

    def enable_save(self):
        self.ui.game_buttonBox.button(QtWidgets.QDialogButtonBox.Save).setEnabled(True)

    def on_save(self):
        self.game.save()
        self.games_model.new_game(self.env_name, os.path.basename(self.game.folder))
        self.close()

    def close(self) -> bool:
        self.game.interrupt()
        return super().close()

#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = GamesController(env_used)
#     # window.show()
#     sys.exit(app.exec_())
