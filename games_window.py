import sys
import os

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QLabel, QPushButton, QHBoxLayout
from games_model import GamesModel
from games_view import GamesView
from Ui_scrollbar_v2 import Ui_MainWindow
from Ui_newGame import Ui_new_game_Dialog
from play_minigrid import main

env_used = 'MiniGrid-Empty-6x6-v0'
games_path = 'games'

folder_name = '2019-12-11_13:40:14'


class MainWindow:

    def __init__(self, env):
        # model
        self.model = GamesModel(env)

        # view (gui)
        self.view = GamesView(env, self.model)
        self.view.show()

        # gui for new game
        self.new_game_view_Dialog = NewGameView()

        # connect model signals to slots
        self.model.new_game_s.connect(self.view.add_row)
        # self.model.new_game_s.connect(self.create_delete_pb_connection)

        self.model.game_removed.connect(self.view.remove_game_from_gui)
        self.model.game_moved.connect(self.view.move_game_gui)

        self.model.moved_up.connect(self.view.move_game_up_gui)
        self.model.moved_down.connect(self.view.move_game_down_gui)

        # connect buttons events to slots (without view)
        # self.ui.new_game_pb.clicked.connect(lambda: self.model.new_game(env_used, 'game ' + str(self.model.n_games)))

        # connect buttons events to slots
        self.view.ui.new_game_pb.clicked.connect(lambda: self.create_new_game(env_used, 'game ' + str(self.model.n_games)))
        # self.view.ui.remove_game_pb.clicked.connect(lambda: self.model.remove_game('games', 0))

        # self.view.ui.games_verticalLayout.itemAt(2).layout().itemAt(4).widget().clicked.connect(lambda: self.view.move_game_gui('games', 2))

        # connect delete btns for games list
        # games_items = self.view.get_items_in_list('games')
        # for item in games_items:
        #     item.layout().itemAt(4).widget().clicked.connect(lambda: self.model.move_game('games', 0))
            # item.layout().indexOf(5)
            # .clicked.connect(self.model.move_game('games', 0))

        # for i, item in enumerate(games_items):
        #     # print(i)
        #     # print(type(item.layout()))
        #     print(type(item.layout().itemAt(3)))
        #     item.layout().itemAt(3).widget().clicked.connect(lambda: self.model.remove_game('games', i))
        #     # print(self.view.ui.games_verticalLayout.indexOf(item.widget()))
        #     print(self.view.ui.games_verticalLayout.indexOf(item.widget()))
        #     print(item.indexOf(item.layout().itemAt(2).widget()))
        #     # .clicked.connect(lambda: self.model.remove_game('games', i))


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

        save = self.new_game_view_Dialog.exec_()
        # TODO: folder = ...
        if save:
            self.model.new_game( env, 'game ' + str(self.model.n_games))
            # self.add_row(env, name, folder_name)

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

class NewGameView(QDialog):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ui = Ui_new_game_Dialog()
        self.ui.setupUi(self)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow(env_used)
    # window.show()
    sys.exit(app.exec_())
