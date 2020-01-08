from datetime import datetime
import sys
import gym
import os

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QLabel, QPushButton, QHBoxLayout
from games_model import GamesModel
from games_view import GamesView
from Ui_scrollbar_v2 import Ui_MainWindow
from Ui_newGame import Ui_new_game_Dialog
from play_minigrid import main
from play_minigrid_one import *

env_used = 'MiniGrid-Empty-6x6-v0'
games_path = 'games'

folder_name = '2019-12-11_13:40:14'


class GamesController:

    def __init__(self, env, agents_model=None):

        # env
        self.env = env

        # agents_model
        self.agents_model = agents_model

        # games_model
        self.games_model = GamesModel(env)

        # view (gui)
        self.view = GamesView(env, self.games_model)
        self.view.show()

        # gui for new game
        self.new_game_view_Dialog = NewGameView(env)

        # connect games_model signals to slots
        self.games_model.new_game_s.connect(self.view.add_row)
        # self.games_model.new_game_s.connect(self.create_delete_pb_connection)

        self.games_model.game_removed.connect(self.view.remove_game_from_gui)
        self.games_model.game_moved.connect(self.view.move_game_gui)

        self.games_model.moved_up.connect(self.view.move_game_up_gui)
        self.games_model.moved_down.connect(self.view.move_game_down_gui)

        # connect buttons events to slots (without view)
        # self.ui.new_game_pb.clicked.connect(lambda: self.games_model.new_game(env_used, 'game ' + str(self.games_model.n_games)))

        # connect buttons events to slots
        self.view.ui.new_game_pb.clicked.connect(lambda: self.create_new_game(env_used, 'game ' + str(self.games_model.n_games)))
        # self.view.ui.remove_game_pb.clicked.connect(lambda: self.games_model.remove_game('games', 0))
        self.view.ui.train_pb.clicked.connect(self.train_agent_slot)

        # self.view.ui.games_verticalLayout.itemAt(2).layout().itemAt(4).widget().clicked.connect(lambda: self.view.move_game_gui('games', 2))

        # connect delete btns for games list
        # games_items = self.view.get_items_in_list('games')
        # for item in games_items:
        #     item.layout().itemAt(4).widget().clicked.connect(lambda: self.games_model.move_game('games', 0))
            # item.layout().indexOf(5)
            # .clicked.connect(self.games_model.move_game('games', 0))

        # for i, item in enumerate(games_items):
        #     # print(i)
        #     # print(type(item.layout()))
        #     print(type(item.layout().itemAt(3)))
        #     item.layout().itemAt(3).widget().clicked.connect(lambda: self.games_model.remove_game('games', i))
        #     # print(self.view.ui.games_verticalLayout.indexOf(item.widget()))
        #     print(self.view.ui.games_verticalLayout.indexOf(item.widget()))
        #     print(item.indexOf(item.layout().itemAt(2).widget()))
        #     # .clicked.connect(lambda: self.games_model.remove_game('games', i))

    # DELETE
    # def initUI(self, env):
    #     """
    #      Main window initialization
    #     :param env: current environment used
    #     :return:
    #     """
    #     # self.ui = Ui_MainWindow()
    #     # self.ui.setupUi(self)
    #     # set window title (env name)
    #     self.setWindowTitle(env)
    #
    #     # self.new_game_Dialog = NewGame()
    #     # self.ui.new_game_pb.clicked.connect(lambda: self.add_row('row1'))
    #
    #     for traj_idx, traj in enumerate(os.listdir(os.path.join(games_path, env))):
    #         # print(traj)
    #         self.add_row(env, 'game ' + str(traj_idx), traj)

    def create_new_game(self, env, name):
        """
        creation of a new game
        :param env: current environment used
        :param name: name for the new game (really needed?)
        :return:
        """
        # TODO play minigrid
        # main()

        # open new game window _done_
        # play
        # save game
        # close

        self.new_game_view_Dialog.play_new_game(env )

        # TODO: folder = ...
        # if save:
        #     self.model.new_game(env, 'game ' + str(self.model.n_games))
        #     # self.add_row(env, name, folder_name)

    # def add_row(self, env, name, folder_name):
    #     """
    #     add row for a new game to games gui
    #     :param env: current environment used
    #     :param name: name for the new game
    #     :param folder_name: name of the new game folder
    #     :return:
    #     """
    #     horiz = QHBoxLayout()
    #     horiz.addWidget(QLabel(name))
    #     # count = 0
    #     img1_path = [elem for elem in os.listdir(os.path.join(games_path, env, folder_name)) if elem.endswith(".png")]
    #     img1_path.sort()
    #     img_path = os.path.join(games_path, env, folder_name, 'game1.png')
    #     # print(img_path)
    #     pixmap = QPixmap(os.path.join(games_path, env, folder_name, 'game1.png'))
    #     # print(path_of_image + '0' + '.png')
    #     label = QLabel()
    #     label.setPixmap(pixmap)
    #     horiz.addWidget(label)
    #     horiz.addWidget(QPushButton('info'))
    #     horiz.addWidget(QPushButton('->'))
    #
    #     self.view.ui.verticalLayout_2.addLayout(horiz)
    #     pass

    def print_(self):
        print('funziona')

    def train_agent_slot(self):
        if self.agents_model is None:
            print("Error: agents_model is None")
            return
        self.agents_model.create_agent(self.env, ["2019-12-17_23:20:25", "2019-12-17_23:19:35", "2019-12-17_23:17:48"]) # TODO cambiareeeeee (ottenere la lista da games_model)
        # TODO: chiudere finestra se agent viene creato correttamente


class NewGameView(QDialog):
    def __init__(self, environment):
        super().__init__()
        self.ui = Ui_new_game_Dialog()
        self.ui.setupUi(self)
        self.env = gym.make(environment)

    def play_new_game(self, environment):
        # minigrid_play_one(env)
        game_label = self.ui.game_label
        state, game_directory, pixmap = self.reset_env(self.env)
        pixmap = self.env.render('pixmap')
        game_label.setPixmap(pixmap)
        pixmap.window.setKeyDownCb(keyDownCb)
        done = False
        while not done:
            self.env.render('pixmap')

            if pixmap.window is None:
                break

    def state_filter(self, state):
        return state['image'][:, :, 0]

    def reset_env(self, env):
        """
        reset the environment, initialize game_name, game_info and directory

        :param env: gym environment used
        :return:
        """
        global game_name, game_info, game_directory, screenshots
        state = self.env.reset()
        self.env.render()
        if hasattr(self.env, 'mission'):
            print('Mission: %s' % self.env.mission)

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
        # TODO: fix all
        game_directory = os.path.join(games_path, self.env, str(game_name))

        screenshot_file = 'game' + str(self.env.step_count) + '.png'
        pixmap = self.env.render('pixmap')
        screenshots = [(screenshot_file, pixmap)]

        return state, game_directory, pixmap



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GamesController(env_used)
    # window.show()
    sys.exit(app.exec_())
