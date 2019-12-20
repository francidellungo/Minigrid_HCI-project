import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QLabel, QPushButton, QHBoxLayout
from PyQt5.QtGui import QIcon, QPixmap, QMovie
from PyQt5.QtCore import QByteArray, QTimer
# from Ui_mainWindow import Ui_MainWindow

from Ui_minigrid import Ui_MainWindow
from Ui_newGame import Ui_new_game_Dialog
from Ui_main_scrollbar import Ui_MainWindow

# class MainWindow_(QMainWindow):
#     def __init__(self):
#         super().__init__()
#
#         self.ui = Ui_MainWindow()
#         self.ui.setupUi(self)
#         pixmap = QPixmap('games/MiniGrid-Empty-6x6-v0/2019-12-11_13:40:14/game1.png')
#
#         self.ui.image_1.setPixmap(pixmap)
#         self.ui.image_2.setPixmap(pixmap)
#         horiz_l1 = QHBoxLayout()
#         label1 = QLabel('label1')
#         b1 = QPushButton('b1')
#         horiz_l1.addWidget(label1)
#         horiz_l1.addWidget(b1)
#
#         self.ui.verticalLayout.addLayout(horiz_l1)
#         # connect button New Game
#         self.new_game_Dialog = NewGame()
#         # self.num_games = 2
#         # self.ui.new_game_button.clicked.connect(lambda: self.new_game_Dialog.exec_())
#         self.ui.new_game_button.clicked.connect(lambda: self.add_row('row1'))
#
#     def add_row(self, name):
#         horiz = QHBoxLayout()
#         label = QLabel(name)
#         info_button = QPushButton('info')
#         move_button = QPushButton('->')
#         horiz.addWidget(label)
#         horiz.addWidget(info_button)
#         horiz.addWidget(move_button)
#         self.ui.verticalLayout_games.addLayout(horiz)


# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#
#         self.ui = Ui_MainWindow()
#         self.ui.setupUi(self)
#         self.games_items = []
#         self.ranking_items = []
#         # for each game create a row and add to the tree
#
#         label = QLabel()
#         label.setText('game 1')
#         button = QPushButton()
#         button.setText('info')
#         image_1 = [label, 'image1', button, 'move ->']
#         strings = ['ciao', 'dadas', 'dddd']
#         l = []  # list of QTreeWidgetItem to add
#         for i in strings:
#             l.append(QTreeWidgetItem([i]))  # create QTreeWidgetItem's and append them
#         # tree.addTopLevelItems(l)
#
#         self.ui.tree_games.addTopLevelItems(QTreeWidgetItem(image_1))
#
#         # item = QTreeWidgetItem()
#         self.ui.tree_games.addTopLevelItems(l)
#

path_of_image = 'games/MiniGrid-Empty-6x6-v0/2019-12-11_13:40:14/game'


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

        # movie = QMovie('image/example.gif', QByteArray())
        # self.ui.image_1.setMovie(movie)
        # movie.start()

        # pixmap = QPixmap('games/MiniGrid-Empty-6x6-v0/2019-12-11_13:40:14/game1.png')
        # self.ui.image_1.setPixmap(pixmap)

        # connect button New Game

        # self.ui.new_game_pb.clicked.connect(lambda: self.new_game_Dialog.exec_())
        # ly = self.ui.games_verticalLayout

        # show sequence of images
        # TODO show images in loop
        # TODO change image only when mouse is on it
        self.count = 0
        timer = QTimer(self)
        timer.timeout.connect(lambda: self.update_image(self.count))
        timer.start(200)
        self.update_image(self.count)

    def initUI(self):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.new_game_Dialog = NewGame()
        self.ui.new_game_pb.clicked.connect(lambda: self.add_row('row1'))


    def update_image(self, count):
        image = path_of_image + str(count) + '.png'
        pixmap = QPixmap(image)
        print('new image : ' + image + ' '+ str(pixmap.isNull()))

        # if not pixmap.isNull():
        self.count = self.count + 1
        self.ui.image_1.setPixmap(pixmap)
        # self.label.adjustSize()
        # self.ui.image_1.resize(pixmap.size())

    def add_row(self, name):
        horiz = QHBoxLayout()
        label = QLabel(name)
        info_button = QPushButton('info')
        move_button = QPushButton('->')
        horiz.addWidget(label)
        horiz.addWidget(info_button)
        horiz.addWidget(move_button)

        self.ui.games_verticalLayout.addLayout(horiz)

class NewGame(QDialog):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ui = Ui_new_game_Dialog()
        self.ui.setupUi(self)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
