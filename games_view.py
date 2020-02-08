import json
import os
import sys
from itertools import cycle

from PyQt5 import QtGui, QtCore

from Ui_scrollbar_v2 import Ui_MainWindow

from PyQt5.QtGui import QPixmap, QColor, QCursor, QIcon, QPalette
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer, pyqtSignal, QPropertyAnimation, pyqtProperty, Qt

env_used = 'MiniGrid-Empty-6x6-v0'
games_path = 'games'

folder_name = '2020-01-05_15:04:54'


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

        self.games_list_view = GamesListView(games_model, env, False, True, False, True, "games")
        self.ui.games_scrollArea.setWidget(self.games_list_view)
        self.ranked_games_list_view = GamesListView(games_model, env, True, False, True, True, "ranked")
        self.ui.game_ranking_scrollArea.setWidget(self.ranked_games_list_view)

        self.setWindowTitle(env)
        self.agents_window = agents_window
        self.ui.train_pb.clicked.connect(self.train_agent_slot)

        for game_key in reversed(self.games_model.games_list):
            self.games_list_view.add_game(game_key)

    def add_row(self, env, game_key):
        self.games_list_view.add_game(game_key)

    def remove_game_from_gui(self, game_key):
        self.games_list_view.remove_game(game_key) or self.ranked_games_list_view.remove_game(game_key)

    def move_game_gui(self, dest_list_name, game_key):
        if dest_list_name == 'games':
            self.ranked_games_list_view.remove_game(game_key)
            self.games_list_view.add_game(game_key, do_anim=True)
        else:
            self.games_list_view.remove_game(game_key)
            self.ranked_games_list_view.add_game(game_key, do_anim=True)

    def move_game_down_gui(self, game_key):
        if self.games_list_view.contains(game_key):
            self.games_list_view.move_game_down_gui(game_key)
        elif self.ranked_games_list_view.contains(game_key):
            self.ranked_games_list_view.move_game_down_gui(game_key)

    def move_game_up_gui(self, game_key):
        if self.games_list_view.contains(game_key):
            self.games_list_view.move_game_up_gui(game_key)
        elif self.ranked_games_list_view.contains(game_key):
            self.ranked_games_list_view.move_game_up_gui(game_key)

    def train_agent_slot(self):
        if self.agents_model is None:
            print("Error: _agents_model is None")
            return
        self.agents_model.create_agent(self.env, self.games_model.ranked_games)
        self.close()
        # print(self.games_model.ranked_games)
        # print(self.ui.ranking_verticalLayout)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        print("closing")  # TODO check why this window sometimes closes by its own and this event it's not raised
        super().closeEvent(a0)
        self.agents_window.ui.btn_create.setEnabled(True)


class GamesListView(QWidget):

    def __init__(self, games_model, env, has_left_arrow=False, has_right_arrow=False, has_updown_arrows=False, has_delete_btn=False, name=None):
        super().__init__()
        self.env = env
        self.games_model = games_model
        self.has_left_arrow = has_left_arrow
        self.has_right_arrow = has_right_arrow
        self.has_updown_arrows = has_updown_arrows
        self.has_delete_btn = has_delete_btn

        self.vertical_layout = QVBoxLayout(self)
        self.games_widgets = {}

        if name is not None:
            self.setObjectName(name)

        self.vertical_layout.addStretch()

    def add_game(self, game_key, position=None, do_anim=False):
        
        if game_key in self.games_widgets:
            return False

        row = GameView(self, game_key, self.games_model, self.env, self.has_left_arrow, self.has_right_arrow, self.has_updown_arrows, self.has_delete_btn)
        self.games_widgets[game_key] = row

        if position is None:
            position = self.vertical_layout.count() - 1
        self.vertical_layout.insertWidget(position, row)

        self.check_enable_btn()

        if do_anim:
            row.doAnim()

        return True

    def remove_game(self, game_key):
        if game_key not in self.games_widgets:
            return False
        
        self._remove_game_from_gui(self.games_widgets[game_key])
        self.games_widgets.pop(game_key)
        
        self.check_enable_btn()
        return True

    def contains(self, game_key):
        return game_key in self.games_widgets

    def _remove_game_from_gui(self, widget_row):
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
                    item.deleteLater()
        widget_row.deleteLater()
        widget_row.parent().layout().removeWidget(widget_row)
        return removed
    
    def check_enable_btn(self):
        if not self.has_updown_arrows:
            return

        # only 1 element in list
        if self.vertical_layout.count() == 2:  # ==2 because of the stretch
            self.vertical_layout.itemAt(0).widget().findChild(QPushButton, 'move_up_btn').setEnabled(False)
            self.vertical_layout.itemAt(0).widget().findChild(QPushButton, 'move_down_btn').setEnabled(False)
            return

        for row_idx in range(self.vertical_layout.count() - 1):  # -1 is for the stretch
            if row_idx == 0:
                self.vertical_layout.itemAt(row_idx).widget().findChild(QPushButton, 'move_up_btn').setEnabled(False)
                self.vertical_layout.itemAt(row_idx).widget().findChild(QPushButton, 'move_down_btn').setEnabled(True)
            elif row_idx == self.vertical_layout.count() - 2:  # -2 because of the stretch
                self.vertical_layout.itemAt(row_idx).widget().findChild(QPushButton, 'move_up_btn').setEnabled(True)
                self.vertical_layout.itemAt(row_idx).widget().findChild(QPushButton, 'move_down_btn').setEnabled(False)
            else:
                self.vertical_layout.itemAt(row_idx).widget().findChild(QPushButton, 'move_up_btn').setEnabled(True)
                self.vertical_layout.itemAt(row_idx).widget().findChild(QPushButton, 'move_down_btn').setEnabled(True)

    def move_game_down_gui(self, game_key):
        old_idx = self.vertical_layout.indexOf(self.findChild(QWidget, game_key))
        self.remove_game(game_key)
        self.add_game(game_key, position=old_idx+1, do_anim=True)

    def move_game_up_gui(self, game_key):
        old_idx = self.vertical_layout.indexOf(self.findChild(QWidget, game_key))
        self.remove_game(game_key)
        self.add_game(game_key, position=old_idx-1, do_anim=True)


class GameView(QWidget):
    
    def __init__(self, parent, game_key, games_model, env, has_left_arrow=False, has_right_arrow=False, has_updown_arrows=False, has_delete_btn=False):
        super().__init__(parent)
        self.game_key = game_key
        self.games_model = games_model
        self.env = env
        self.has_left_arrow = has_left_arrow
        self.has_right_arrow = has_right_arrow
        self.has_updown_arrows = has_updown_arrows
        self.has_delete_btn = has_delete_btn
        self.setObjectName(game_key)
    
        self.horiz = QHBoxLayout(self)
        self.default_color = self.palette().color(QtGui.QPalette.Background)
        self.hover_color = Qt.lightGray
        self.setAutoFillBackground(True)
        self.is_deleted = False
        
        # move left btn
        if self.has_left_arrow:
            self.move_left_btn = QPushButton()
            #self.move_left_btn.setIcon(QIcon('img/arrowLeft.png'))
            self.move_left_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowLeft))
            self.move_left_btn.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
            self.horiz.addWidget(self.move_left_btn)
            self.move_left_btn.clicked.connect(lambda: self.games_model.move_game(self.parent().objectName(), game_key))

        self.game_label = QLabel(game_key)
        game_info_path = os.path.join(games_path, self.env, game_key, 'game.json')
        with open(game_info_path) as json_file:
            game_info = json.load(json_file)
            
        self.game_label.setWordWrap(True)
        self.str_game_short_label = str(game_info['name'])
        self.str_game_long_label = "Name: " + str(game_info['name']) + "\nCreated: " + game_key + "\n# actions: " + str(len(game_info['trajectory']) - 1)
        self.game_label.setText(self.str_game_short_label)
        self.horiz.addWidget(self.game_label)

        # game imgs
        imgs_names = [elem for elem in os.listdir(os.path.join(games_path, env, game_key)) if elem.endswith(".png")]
        imgs_names.sort()
        self.game_dir = os.path.join(games_path, env, game_key)
        pixmap = QPixmap(os.path.join(games_path, env, game_key, 'game0.png'))
        self.img_label = QLabel()
        self.img_label.setScaledContents(True)
        self.img_label.setPixmap(pixmap)
        self.img_label.setFixedSize(150, 150)
        self.timer = QTimer()

        self.horiz.addWidget(self.img_label)

        if has_updown_arrows or has_delete_btn:
            self.btns_group = QWidget()
            self.btns_group_layout = QVBoxLayout(self.btns_group)

            # move up/down buttons
            if has_updown_arrows:
                self.move_up_btn = QPushButton(self)
                self.move_up_btn.setObjectName('move_up_btn')
                #self.move_up_btn.setIcon(QIcon('img/arrowUp.png'))
                self.move_up_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowUp))
                self.move_up_btn.setFixedWidth(30)
                self.move_down_btn = QPushButton(self)
                self.move_down_btn.setObjectName('move_down_btn')
                #self.move_down_btn.setIcon(QIcon('img/arrowDown.png'))
                self.move_down_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowDown))
                self.move_down_btn.setFixedWidth(30)

                self.btns_group_layout.addWidget(self.move_up_btn, alignment=QtCore.Qt.AlignCenter)

                self.move_up_btn.clicked.connect(lambda: self.games_model.move_down(game_key))
                self.move_down_btn.clicked.connect(lambda: self.games_model.move_up(game_key))

            # delete game btn
            if has_delete_btn:
                self.delete_pb = QPushButton()
                self.delete_pb.setIcon(self.style().standardIcon(QStyle.SP_DialogCloseButton))

                self.delete_pb.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
                self.btns_group_layout.addWidget(self.delete_pb, alignment=QtCore.Qt.AlignCenter)
                self.delete_pb.clicked.connect(lambda: self.confirm_delete_dialog())

            if has_updown_arrows:
                self.btns_group_layout.addWidget(self.move_down_btn, alignment=QtCore.Qt.AlignCenter)

            self.btns_group.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
            self.horiz.addWidget(self.btns_group)

        # move right btn
        if has_right_arrow:
            self.move_right_btn = QPushButton()
            self.move_right_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
            # self.move_right_btn.setIcon(QIcon('img/arrowRight.png'))
            self.move_right_btn.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
            self.horiz.addWidget(self.move_right_btn)
            self.move_right_btn.clicked.connect(lambda: self.games_model.move_game(self.parent().objectName(), game_key))

        # reinizialize image
        imgs_nums = self.get_imgs_nums()
        imgs_nums.sort(key=int)
        imgs = ['game' + str(img_num) + '.png' for img_num in imgs_nums]
        self.default_img = imgs[0]

    def _set_color(self, col):
        palette = self.palette()
        palette.setColor(self.backgroundRole(), col)
        self.setPalette(palette)

    color = pyqtProperty(QColor, fset=_set_color)

    def confirm_delete_dialog(self):
        button_reply = QMessageBox.question(self, 'Confirm deletion', "Are you sure you want to delete " + str(self.objectName() + " ?"),
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if button_reply == QMessageBox.Yes:
            self.games_model.remove_game(self.game_key)

    def get_imgs_nums(self):
        return [elem.split('game')[1].split('.')[0] for elem in os.listdir(self.game_dir) if elem.endswith(".png")]

    def show_traj_imgs(self):
        imgs_nums = self.get_imgs_nums()
        imgs_nums.sort(key=int)
        imgs = ['game' + str(img_num) + '.png' for img_num in imgs_nums]

        imgs_cycle = cycle(imgs)

        # show images in loop with only one timer.
        self.timer.timeout.connect(lambda: self.on_timeout(os.path.join(self.game_dir, next(imgs_cycle))))
        self.timer.start(250)

    def stop_show_traj(self):
        self.timer.stop()
        self.on_timeout(os.path.join(self.game_dir, self.default_img))

    def on_timeout(self, image):
        try:
            pixmap = QPixmap(image)
            if not pixmap.isNull():
                try:
                    self.img_label.setPixmap(pixmap)
                except RuntimeError:
                    self.timer.stop()
        except StopIteration:
            pass

    def doAnim(self):
        self.anim = QPropertyAnimation(self, b"color")
        row_color = self.palette().color(QtGui.QPalette.Background).name()
        self.anim.setDuration(1000)
        self.anim.setStartValue(QColor('gray'))
        self.anim.setEndValue(QColor(row_color))
        self.anim.start()

    def up_arrow_click(self):
        pass

    def down_arrow_click(self):
        pass

    def enterEvent(self, a0: QtCore.QEvent) -> None:
        super().enterEvent(a0)
        self._set_color(self.hover_color)
        self.game_label.setText(self.str_game_long_label)
        self.show_traj_imgs()

    def leaveEvent(self, a0: QtCore.QEvent) -> None:
        super().leaveEvent(a0)
        self._set_color(self.default_color)
        self.game_label.setText(self.str_game_short_label)
        self.stop_show_traj()

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = GamesView(env_used)
#     window.show()
#     sys.exit(app.exec_())
