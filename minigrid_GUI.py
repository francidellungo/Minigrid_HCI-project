import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, qApp
from PyQt5.QtGui import QIcon, QPixmap

from Ui_minigrid import Ui_MainWindow
from Ui_newGame import Ui_new_game_Dialog


class MainWindow_(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        pixmap = QPixmap('games/MiniGrid-Empty-6x6-v0/2019-12-11_13:40:14/game1.png')

        self.ui.image_1.setPixmap(pixmap)
        self.ui.image_2.setPixmap(pixmap)

        # connect button New Game
        self.new_game_Dialog = NewGame()
        # self.num_games = 2
        self.ui.new_game_button.clicked.connect(lambda: self.new_game_Dialog.exec_())


class NewGame(QDialog):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ui = Ui_new_game_Dialog()
        self.ui.setupUi(self)





if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow_()
    window.show()
    sys.exit(app.exec_())
