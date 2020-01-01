from PyQt5.QtGui import QPixmap
import os
import sys
from Ui_scrollbar_v2 import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QLabel, QPushButton, QHBoxLayout

env_used = 'MiniGrid-Empty-6x6-v0'
games_path = 'games'

folder_name = '2019-12-11_13:40:14'


class GamesView(QMainWindow):

    # list of

    def __init__(self, env):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.initUI(env)
        self.setWindowTitle(env)


    # def add_row(self, name):
    #     horiz = QHBoxLayout()
    #     horiz.addWidget(QLabel(name))
    #     count = 0
    #     # pixmap = QPixmap(path_of_image + '0' + '.png')
    #     pixmap = QPixmap('games/MiniGrid-Empty-6x6-v0/2019-12-11_13:40:14/game1.png')
    #     # print(path_of_image + '0' + '.png')
    #     label = QLabel()
    #     label.setPixmap(pixmap)
    #     horiz.addWidget(label)
    #     horiz.addWidget(QPushButton('info'))
    #     horiz.addWidget(QPushButton('->'))
    #
    #     self.ui.verticalLayout_2.addLayout(horiz)
    #     # self.update_image(0)

    def add_row(self, env, name, folder_name):
        """
        add row for a new game to games gui
        :param env: current environment used
        :param name: name for the new game
        :param folder_name: name of the new game folder
        :return:
        """
        horiz = QHBoxLayout()
        horiz.addWidget(QLabel(name))
        img1_path = [elem for elem in os.listdir(os.path.join(games_path, env, folder_name)) if elem.endswith(".png")]
        img1_path.sort()
        # img_path = os.path.join(games_path, env, folder_name, 'game1.png')
        # print(img_path)
        pixmap = QPixmap(os.path.join(games_path, env, folder_name, 'game1.png'))
        # print(path_of_image + '0' + '.png')

        label = QLabel()
        label.setPixmap(pixmap)
        horiz.addWidget(label)

        horiz.addWidget(QPushButton('info'))
        horiz.addWidget(QPushButton('delete'))
        horiz.addWidget(QPushButton('->'))
        # self.horizLayouts_g.append(horiz)
        self.ui.games_verticalLayout.addLayout(horiz)

    def initUI(self, env):
        """
         Main window initialization
        :param env: current environment used
        :return:
        """
        # self.ui = Ui_MainWindow()
        # self.ui.setupUi(self)
        # set window title (env name)
        self.setWindowTitle(env)

        # self.new_game_Dialog = NewGame()
        # self.ui.new_game_pb.clicked.connect(lambda: self.add_row('row1'))

        for traj_idx, traj in enumerate(os.listdir(os.path.join(games_path, env))):
            # print(traj)
            self.add_row(env, 'game ' + str(traj_idx), traj)

    def remove_game_from_gui(self, current_list, game_idx):
        """
        remove selected game from the list
        :param current_list: 'games' if the game is in games list, 'rank' if game is in ranking list
        :param game_idx: index of the game in the list where it is
        :return:
        """
        # open window are you sure you want to delete?
        # then remove item from the list (and put to_delete = True)
        # layout_to_remove = self.horizLayouts_g[game_idx] if current_list == 'games' else self.horizLayouts_r[game_idx]
        # TODO control if idx is not out of range
        l_item = self.ui.games_verticalLayout.itemAt(game_idx).layout() if current_list == 'games' else self.ui.ranking_verticalLayout.itemAt(game_idx).layout()
        self.clear_layout(l_item)

    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clearLayout(item.layout())

    def move_game_gui(self, starting_list, game_idx_start_list):
        pass

    def move_game_up_gui(self, game_idx):
        pass

    def move_game_down_gui(self, game_idx):
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GamesView(env_used)
    window.show()
    sys.exit(app.exec_())
