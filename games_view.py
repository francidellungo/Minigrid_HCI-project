import json
import os
import sys
from itertools import cycle

from PyQt5 import QtGui

from Ui_scrollbar_v2 import Ui_MainWindow

from PyQt5.QtGui import QPixmap, QColor, QCursor, QIcon, QPalette
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QLabel, QPushButton, QHBoxLayout, QWidget, QMessageBox, \
    QGraphicsColorizeEffect, QGraphicsOpacityEffect
from PyQt5.QtCore import QTimer, pyqtSignal, QPropertyAnimation, pyqtProperty

env_used = 'MiniGrid-Empty-6x6-v0'
games_path = 'games'

folder_name = '2020-01-05_15:04:54'

bg_color = QColor('#efebe7')


class MyWidget(QWidget):

    def __init__(self, parent):
        super().__init__(parent)

    def _set_color(self, col):
        palette = self.palette()
        palette.setColor(self.backgroundRole(), col)
        self.setPalette(palette)

    color = pyqtProperty(QColor, fset=_set_color)


class GamesView(QMainWindow):

    def __init__(self, env, games_model, agents_window, agents_model):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.games_model = games_model
        self.agents_model = agents_model
        self.env = env
        # self.counts = []
        self.animations = []
        self.bg_color = self.palette().color(QPalette.Background)

        self.initUI(env)
        self.setWindowTitle(env)
        self.agents_window = agents_window
        self.ui.train_pb.clicked.connect(self.train_agent_slot)

    def add_row(self, env, game_key, list_='games', position=None):
        """
        add row for a new game to games gui
        :param position: index of the position to insert the new row in the layout list
        :param list_: 'games' or 'rank'
        :param env: current environment used
        :param name: name for the new game
        :param game_key: name of the new game folder
        :return:
        """
        if list_ is 'games':
            row = MyWidget(self.ui.games_verticalW)
        else:
            row = MyWidget(self.ui.ranking_verticalW)
        row.setAutoFillBackground(True)

        row.setObjectName(game_key)
        horiz = QHBoxLayout(row)

        # move btn case list 'rank'
        if list_ == 'rank':
            move_btn = QPushButton('<-')
            horiz.addWidget(move_btn)

        # horiz = QHBoxLayout()
        # game name
        game_label = QLabel(game_key)
        game_info_path = os.path.join(games_path, self.env, game_key, 'game.json')
        with open(game_info_path) as json_file:
            game_info = json.load(json_file)

        tooltip_msg = 'Game: ' + str(game_info['name']) + '\n# steps: ' + str(len(game_info['trajectory'])-1) + '\nScore: ' + str(game_info['score'])
        game_label.setToolTip(tooltip_msg)
        horiz.addWidget(game_label)

        # game imgs
        imgs_names = [elem for elem in os.listdir(os.path.join(games_path, env, game_key)) if elem.endswith(".png")]
        imgs_names.sort()
        dir_path = os.path.join(games_path, env, game_key)

        print("adding row for game {}, num_states: {}".format(game_key, len(self.get_imgs_nums(dir_path))))

        pixmap = QPixmap(os.path.join(games_path, env, game_key, 'game0.png'))

        img_label = QLabel()
        img_label.setScaledContents(True)
        img_label.setPixmap(pixmap)
        img_label.setFixedSize(150, 150)

        timer = QTimer()

        # show trajectories in loop when pass on it with mouse
        img_label.enterEvent = lambda ev: self.show_traj_imgs(dir_path, img_label, timer)
        img_label.leaveEvent = lambda ev: self.stop_show_traj(dir_path, img_label, timer)

        horiz.addWidget(img_label)

        # game info button
        # horiz.addWidget(QPushButton('info'))

        # delete game button
        delete_pb = QPushButton('delete')
        horiz.addWidget(delete_pb)

        # move game button case insertion in list 'games'
        if list_ == 'games':
            move_btn = QPushButton('->')
            horiz.addWidget(move_btn)
        else:
            # in 'rank' list
            move_up_btn = QPushButton()
            move_up_btn.setObjectName('move_up_btn')
            move_up_btn.setIcon(QIcon('img/arrowUp.jpeg'))
            move_up_btn.setFixedWidth(30)
            move_down_btn = QPushButton()
            move_down_btn.setObjectName('move_down_btn')
            move_down_btn.setIcon(QIcon('img/arrowDown.jpeg'))
            move_down_btn.setFixedWidth(30)
            horiz.addWidget(move_up_btn)
            horiz.addWidget(move_down_btn)

        # add row to vertical layout
        if list_ == 'games':
            self.ui.games_verticalLayout.addWidget(row)

        else:
            if position is None:
                self.ui.ranking_verticalLayout.addWidget(row)
            else:
                self.ui.ranking_verticalLayout.insertWidget(position, row)

        # connect all the buttons:

        # delete game push button
        delete_pb.clicked.connect(lambda: self.confirm_delete_dialog(row))

        # move game between lists push button
        move_btn.clicked.connect(lambda: self.games_model.move_game(('games' if row.parent().objectName() == 'games_verticalW' else 'rank'), row.objectName()))

        if list_ == 'rank':
            # connect move up and down buttons (in ranking list)
            move_up_btn.clicked.connect(lambda: self.games_model.move_down(game_key))
            move_up_btn.setEnabled(True)

            move_down_btn.clicked.connect(lambda: self.games_model.move_up(game_key))
            move_down_btn.setEnabled(True)

        self.check_enable_btn()
        return row

    def check_enable_btn(self):
        # only 1 element in ranking list
        if self.ui.ranking_verticalLayout.count() == 1:
            self.ui.ranking_verticalLayout.itemAt(0).widget().findChild(QPushButton, 'move_up_btn').setEnabled(False)
            self.ui.ranking_verticalLayout.itemAt(0).widget().findChild(QPushButton, 'move_down_btn').setEnabled(False)
            return

        for row_idx in range(self.ui.ranking_verticalLayout.count()):
            if row_idx == 0:
                self.ui.ranking_verticalLayout.itemAt(row_idx).widget().findChild(QPushButton, 'move_up_btn').setEnabled(False)
                self.ui.ranking_verticalLayout.itemAt(row_idx).widget().findChild(QPushButton, 'move_down_btn').setEnabled(True)
            elif row_idx == self.ui.ranking_verticalLayout.count() - 1:
                self.ui.ranking_verticalLayout.itemAt(row_idx).widget().findChild(QPushButton, 'move_up_btn').setEnabled(True)
                self.ui.ranking_verticalLayout.itemAt(row_idx).widget().findChild(QPushButton, 'move_down_btn').setEnabled(False)
            else:
                self.ui.ranking_verticalLayout.itemAt(row_idx).widget().findChild(QPushButton, 'move_up_btn').setEnabled(True)
                self.ui.ranking_verticalLayout.itemAt(row_idx).widget().findChild(QPushButton, 'move_down_btn').setEnabled(True)

    def confirm_delete_dialog(self, row):
        button_reply = QMessageBox.question(self, 'Confirm deletion', "Are you sure you want to delete " + str(row.objectName() + " ?"),
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if button_reply == QMessageBox.Yes:
            self.games_model.remove_game(row.objectName(),
                                         ('games' if row.parent() == self.ui.games_verticalW else 'rank'))
            # print('Yes clicked.')
        # else:
            # print('No clicked.')

    def initUI(self, env):
        """
         Main window initialization
        :param env: current environment used
        :return:
        """
        # set window title (env name)
        self.setWindowTitle(env)

        for traj in reversed(self.games_model.games_list):
            self.add_row(env, traj)

    def remove_game_from_gui(self, folder_name):
        """
        remove selected game from the list
        :param list_: 'games' if the game is in games list, 'rank' if game is in ranking list
        :param folder_name:
        :return:
        """
        # TODO check to_delete and delete
        # remove item from the list (and put to_delete = True)

        w_item = self.ui.centralwidget.findChild(QWidget, folder_name)
        # .findChild(QHBoxLayout)
        # l_item = self.ui.games_verticalLayout.itemAt(game_idx).layout() if current_list == 'games' else self.ui.ranking_verticalLayout.itemAt(game_idx).layout()
        # l_item = self.game_layouts[game_idx]
        # print(self.ui.games_verticalLayout.count(), len(self.game_layouts))
        # print(w_item.objectName())
        removed_widgets = self.clear_layout(w_item)

        return removed_widgets

    def clear_layout(self, widget_row):
        # self.game_layouts.remove(layout)
        layout = widget_row.findChild(QHBoxLayout)
        removed = []
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    removed.append(widget)
                    widget.deleteLater()
                else:
                    self.clearLayout(item.layout())
                    print(item.layout())
        widget_row.deleteLater()
        widget_row.parent().layout().removeWidget(widget_row)
        return removed

    def move_game_gui(self, dest_list_name, game_name):
        self.remove_game_from_gui(game_name)

        row = self.add_row(self.env, game_name, dest_list_name)
        self.doAnim(row)

    def move_game_up_gui(self, game_name):
        #insertItem(index, a1)
        # self.ui.games_verticalLayout.insertLayout()
        # replaceWidget()
        old_idx = self.ui.ranking_verticalLayout.indexOf(self.ui.ranking_verticalW.findChild(QWidget, game_name))
        # print('position: ', old_idx, 'position to insert: ', old_idx - 1)
        self.remove_game_from_gui(game_name)
        row = self.add_row(self.env, game_name, list_='rank', position=old_idx - 1)
        self.doAnim(row)

    def move_game_down_gui(self, game_name):
        old_idx = self.ui.ranking_verticalLayout.indexOf(self.ui.ranking_verticalW.findChild(QWidget, game_name))
        # print('position: ', old_idx, 'position to insert: ', old_idx + 1)
        self.remove_game_from_gui(game_name)
        row = self.add_row(self.env, game_name, list_='rank', position=old_idx + 1)
        self.doAnim(row)

    def doAnim(self, row):
        global bg_color
        anim = QPropertyAnimation(row, b"color")
        anim.setDuration(1000)
        anim.setStartValue(QColor('gray'))
        anim.setEndValue(QColor('#efebe7'))
        anim.start()
        self.animations.append(anim)
        anim.finished.connect(lambda a=anim: self.animations.remove(a))


    def train_agent_slot(self):
        if self.agents_model is None:
            print("Error: _agents_model is None")
            return
        self.agents_model.create_agent(self.env, self.games_model.ranked_games)
        self.close()
        # print(self.games_model.ranked_games)
        # print(self.ui.ranking_verticalLayout)

    def get_imgs_nums(self, games_dir):
        return [elem.split('game')[1].split('.')[0] for elem in os.listdir(games_dir) if elem.endswith(".png")]

    def show_traj_imgs(self, games_dir, img_label, timer):
        imgs_nums = self.get_imgs_nums(games_dir)
        imgs_nums.sort(key=int)
        imgs = ['game' + str(img_num) + '.png' for img_num in imgs_nums]

        imgs_cycle = cycle(imgs)

        # show images in loop with only one timer.
        timer.timeout.connect(lambda: self.on_timeout(os.path.join(games_dir, next(imgs_cycle)), img_label))
        timer.start(250)

    def stop_show_traj(self, games_dir, img_label, timer):
        timer.stop()
        # reinizialize image
        imgs_nums = [elem.split('game')[1].split('.')[0] for elem in os.listdir(games_dir) if elem.endswith(".png")]
        imgs_nums.sort(key=int)
        imgs = ['game' + str(img_num) + '.png' for img_num in imgs_nums]
        self.on_timeout(os.path.join(games_dir, imgs[0]), img_label)

    def on_timeout(self, image, img_label):
        try:
            pixmap = QPixmap(image)
            if not pixmap.isNull():
                img_label.setPixmap(pixmap)
        except StopIteration:
            pass
        #     self.timer.stop()

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        super().closeEvent(a0)
        self.agents_window.ui.btn_create.setEnabled(True)

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = GamesView(env_used)
#     window.show()
#     sys.exit(app.exec_())
