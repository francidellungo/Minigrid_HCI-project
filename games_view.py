import os
import sys
from itertools import cycle

from PyQt5 import QtGui

from Ui_scrollbar_v2 import Ui_MainWindow

from PyQt5.QtGui import QPixmap, QColor, QCursor, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QLabel, QPushButton, QHBoxLayout, QWidget, QMessageBox
from PyQt5.QtCore import QTimer, pyqtSignal

env_used = 'MiniGrid-Empty-6x6-v0'
games_path = 'games'

folder_name = '2020-01-05_15:04:54'

# NOT USED: DELETE
# class ClickLabel(QLabel):
#     clicked = pyqtSignal()
#
#     def mousePressEvent(self, event):
#         self.clicked.emit()
#         QLabel.mousePressEvent(self, event)

# class ClickableLabel(QLabel):
#
#     clicked = pyqtSignal(QLabel)
#     def __init__(self, parent=None):
#         super(ClickableLabel, self).__init__(parent)
#         self.setMouseTracking(True)
#
#     # def mouseMoveEvent(self, event):
#     #     print("On Hover", event.pos().x(), event.pos().y()) # event.pos().x(), event.pos().y()
#
#     def mousePressEvent(self, event):
#         print('mousePressEvent', event)
#         self.clicked.emit(self)


class GamesView(QMainWindow):

    def __init__(self, env, games_model, agents_window, agents_model):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.games_model = games_model
        self.agents_model = agents_model
        self.env = env
        self.counts = []

        self.initUI(env)
        self.setWindowTitle(env)
        self.agents_window = agents_window
        self.ui.train_pb.clicked.connect(self.train_agent_slot)

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

    def add_row(self, env, folder_name, list_='games', position=None):
        """
        add row for a new game to games gui
        :param position: index of the position to insert the new row in the layout list
        :param list_: 'games' or 'rank'
        :param env: current environment used
        :param name: name for the new game
        :param folder_name: name of the new game folder
        :return:
        """
        if list_ is 'games':
            row = QWidget(self.ui.games_verticalW)
        else:
            row = QWidget(self.ui.ranking_verticalW)

        row.setObjectName(folder_name)
        horiz = QHBoxLayout(row)

        # move btn case list 'rank'
        if list_ == 'rank':
            move_btn = QPushButton('<-')
            horiz.addWidget(move_btn)

        # horiz = QHBoxLayout()
        # game name
        horiz.addWidget(QLabel(folder_name))

        # game imgs
        imgs_names = [elem for elem in os.listdir(os.path.join(games_path, env, folder_name)) if elem.endswith(".png")]
        imgs_names.sort()
        dir_path = os.path.join(games_path, env, folder_name)
        # img_path = os.path.join(games_path, env, folder_name, 'game1.png')
        pixmap = QPixmap(os.path.join(games_path, env, folder_name, 'game0.png'))

        img_label = QLabel()
        img_label.setPixmap(pixmap)

        timer = QTimer()

        # img_label.enterEvent = lambda ev: self.show_traj_imgs(dir_path, img_label, timer, self.counts.index(counter))

        # show trajectories in loop when pass on it with mouse
        img_label.enterEvent = lambda ev: self.show_traj_imgs(dir_path, img_label, timer)
        img_label.leaveEvent = lambda ev: self.stop_show_traj(dir_path, img_label, timer)

        horiz.addWidget(img_label)

        # game info button
        horiz.addWidget(QPushButton('info'))

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
            # game_idx = [item['layout'] for item in self.game_layouts if item['layout'] == horiz]

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
            move_up_btn.clicked.connect(lambda: self.games_model.move_down(folder_name))
            move_up_btn.setEnabled(True)
            # move_down_btn.clicked.connect(self.check_enable_btn)

            move_down_btn.clicked.connect(lambda: self.games_model.move_up(folder_name))
            # move_down_btn.clicked.connect(self.check_enable_btn)
            move_down_btn.setEnabled(True)
        self.check_enable_btn()

    def check_enable_btn(self):
        # only 1 element in ranking list
        if self.ui.ranking_verticalLayout.count() == 1:
            self.ui.ranking_verticalLayout.itemAt(0).widget().findChild(QPushButton, 'move_up_btn').setEnabled(False)
            self.ui.ranking_verticalLayout.itemAt(0).widget().findChild(QPushButton, 'move_down_btn').setEnabled(False)
            return

        for row_idx in range(self.ui.ranking_verticalLayout.count()):
            # print(row_idx)
            if row_idx == 0:
                self.ui.ranking_verticalLayout.itemAt(row_idx).widget().findChild(QPushButton, 'move_up_btn').setEnabled(False)
                self.ui.ranking_verticalLayout.itemAt(row_idx).widget().findChild(QPushButton, 'move_down_btn').setEnabled(True)
            elif row_idx == self.ui.ranking_verticalLayout.count() - 1:
                self.ui.ranking_verticalLayout.itemAt(row_idx).widget().findChild(QPushButton, 'move_up_btn').setEnabled(True)
                self.ui.ranking_verticalLayout.itemAt(row_idx).widget().findChild(QPushButton, 'move_down_btn').setEnabled(False)
            else:
                self.ui.ranking_verticalLayout.itemAt(row_idx).widget().findChild(QPushButton, 'move_up_btn').setEnabled(True)
                self.ui.ranking_verticalLayout.itemAt(row_idx).widget().findChild(QPushButton, 'move_down_btn').setEnabled(True)

        # self.ui.ranking_verticalW.findChild(QPushButton, 'move_up_btn').setEnabled(False)

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
        # self.ui = Ui_MainWindow()
        # set window title (env name)
        self.setWindowTitle(env)
        # self.ui.games_verticalLayout.setObjectName('games_verticalLayout')
        # self.ui.ranking_verticalLayout.setObjectName('ranking_verticalLayout')

        # self.new_game_Dialog = NewGame()
        # self.ui.new_game_pb.clicked.connect(lambda: self.add_row('row1'))

        for traj in reversed(self.games_model.games_list):
            self.add_row(env, traj)


    def remove_game_from_gui(self, folder_name):
        """
        remove selected game from the list
        :param list_: 'games' if the game is in games list, 'rank' if game is in ranking list
        :param folder_name:
        :return:
        """
        # open window are you sure you want to delete?
        # then remove item from the list (and put to_delete = True)

        w_item = self.ui.centralwidget.findChild(QWidget, folder_name)
        # .findChild(QHBoxLayout)
        # l_item = self.ui.games_verticalLayout.itemAt(game_idx).layout() if current_list == 'games' else self.ui.ranking_verticalLayout.itemAt(game_idx).layout()
        # l_item = self.game_layouts[game_idx]
        # print(self.ui.games_verticalLayout.count(), len(self.game_layouts))
        # print(w_item.objectName())
        removed_widgets = self.clear_layout(w_item)
        # print('removed_widgets: ', removed_widgets)

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

        # print('game_name: ', game_name, 'dest list: ', dest_list_name)
        self.remove_game_from_gui(game_name)

        self.add_row(self.env, game_name, dest_list_name)
        # self.check_enable_btn()

        # l_item = self.ui.centralwidget.findChild(QWidget, game_name)
        # # l_item.setParent(None)
        # print(l_item.parent().objectName())
        # l_item.setParent(self.ui.ranking_verticalW) if l_item.parent().objectName() == \
        #                                                self.ui.games_verticalW.objectName() else l_item.setParent(self.ui.games_verticalW)

        # cur_list = self.ui.games_verticalLayout if starting_list == 'games' else self.ui.ranking_verticalLayout
        # removed_widgets = self.remove_game_from_gui(cur_list, game_idx_start_list)
        # print(removed_widgets)
        # dest_list = 'games' if starting_list == 'rank' else 'games'
        # self.add_row(env_used, 'ff', folder_name, 'rank')
        # self.ui.games_verticalLayout.removeItem()

    def move_game_up_gui(self, game_name):
        #insertItem(index, a1)
        # self.ui.games_verticalLayout.insertLayout()
        # replaceWidget()
        old_idx = self.ui.ranking_verticalLayout.indexOf(self.ui.ranking_verticalW.findChild(QWidget, game_name))
        # print('position: ', old_idx, 'position to insert: ', old_idx - 1)
        self.remove_game_from_gui(game_name)
        self.add_row(self.env, game_name, list_='rank', position=old_idx - 1)
        # self.check_enable_btn()

    def move_game_down_gui(self, game_name):
        old_idx = self.ui.ranking_verticalLayout.indexOf(self.ui.ranking_verticalW.findChild(QWidget, game_name))
        # print('position: ', old_idx, 'position to insert: ', old_idx + 1)
        self.remove_game_from_gui(game_name)
        self.add_row(self.env, game_name, list_='rank', position=old_idx + 1)
        # self.check_enable_btn()

    # def get_items_in_list(self, list ='games'):
    #     if list == 'games':
    #         games_items = (self.ui.games_verticalLayout.itemAt(i) for i in range(self.ui.games_verticalLayout.count()))
    #     else:
    #         games_items = (self.ui.ranking_verticalLayout.itemAt(i) for i in range(self.ui.ranking_verticalLayout.count()))
    #     return games_items

    def train_agent_slot(self):
        if self.agents_model is None:
            print("Error: _agents_model is None")
            return
        self.agents_model.create_agent(self.env, self.games_model.ranked_games)
        self.close()
        print(self.games_model.ranked_games)
        print(self.ui.ranking_verticalLayout)

    def show_traj_imgs(self, games_dir, img_label, timer):
        imgs_nums = [elem.split('game')[1].split('.')[0] for elem in os.listdir(games_dir) if elem.endswith(".png")]
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
