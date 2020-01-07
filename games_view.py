import os
import sys
from itertools import cycle

from Ui_scrollbar_v2 import Ui_MainWindow

from PyQt5.QtGui import QPixmap, QColor, QCursor
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QLabel, QPushButton, QHBoxLayout
from PyQt5.QtCore import QTimer, pyqtSignal

env_used = 'MiniGrid-Empty-6x6-v0'
games_path = 'games'

folder_name = '2020-01-05_15:04:54'


class ClickLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        self.clicked.emit()
        QLabel.mousePressEvent(self, event)

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

    def __init__(self, env, model):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.model = model
        self.counts = []
        self.game_layouts = []
        # TODO rank_layouts
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

    # TODO fix: add model
    def add_row(self, env, name, folder_name, list_= 'games'):
        """
        add row for a new game to games gui
        :param env: current environment used
        :param name: name for the new game
        :param folder_name: name of the new game folder
        :return:
        """
        horiz = QHBoxLayout()
        # game name
        horiz.addWidget(QLabel(name))

        # game imgs
        imgs_names = [elem for elem in os.listdir(os.path.join(games_path, env, folder_name)) if elem.endswith(".png")]
        imgs_names.sort()
        dir_path = os.path.join(games_path, env, folder_name)
        # img_path = os.path.join(games_path, env, folder_name, 'game1.png')
        pixmap = QPixmap(os.path.join(games_path, env, folder_name, 'game1.png'))
        # print(path_of_image + '0' + '.png')

        label = QLabel()
        label.setPixmap(pixmap)
        label.setMouseTracking(True)

        # show trajectories in loop when click on it
        img_label = ClickLabel()
        img_label.setPixmap(pixmap)
        # horiz.addWidget(label)
        timer = QTimer()
        counter = 1
        self.counts.append(counter)
        self.counts.index(counter)
        print('self.counts.index(self.counter)', self.counts.index(counter))
        img_label.clicked.connect(lambda: self.show_traj_imgs(dir_path, img_label, timer, self.counts.index(counter)))

        horiz.addWidget(img_label)

        # game info button
        horiz.addWidget(QPushButton('info'))

        # delete game button
        delete_pb = QPushButton('delete')
        horiz.addWidget(delete_pb)
        # TODO fix
        # delete_pb.clicked.connect(lambda: self.model.move_game('games', 0))
        idx = 0
        sender = self.sender()
        # delete_pb.clicked.connect(lambda: self.model.remove_game('games', idx))

        # move game button
        move_btn = QPushButton('->')
        horiz.addWidget(move_btn)
        # self.horizLayouts_g.append(horiz)

        if list_ == 'games':
            self.game_layouts.append({'layout': horiz, 'game_folder': folder_name})
            self.ui.games_verticalLayout.addLayout(self.game_layouts[-1]['layout'])


            print(len(self.game_layouts))
            game_idx = [item['layout'] for item in self.game_layouts if item['layout'] == horiz]
            print(game_idx, horiz)
            # [item['layout'] for item in self.game_layouts if item["layout"] == horiz]
            # print('index: ', self.game_layouts.index(game_idx))

            delete_pb.clicked.connect(lambda: self.model.remove_game('games', self.game_layouts.index(horiz)))
            # self.game_layouts.index(item for item in self.game_layouts if item["layout"] == horiz)
            
            # TODO fix  (this is just for games list)

        else:
            self.ui.ranking_verticalLayout.addLayout(horiz)

        move_btn.clicked.connect(lambda: self.model.move_game('games', self.game_layouts.index(horiz)))

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

        l_item = self.game_layouts[game_idx]
        print(self.ui.games_verticalLayout.count(), len(self.game_layouts))
        removed_widgets = self.clear_layout(l_item)

        # print(l_item)
        print(self.ui.games_verticalLayout.count(), len(self.game_layouts))
        return removed_widgets

    def clear_layout(self, layout):
        self.game_layouts.remove(layout)
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
        return removed


    def add_layout(self, layout, list_):
        curr_list = self.ui.games_verticalLayout if list_ == 'games' else self.ui.ranking_verticalLayout
        curr_list.addLayout(layout)

    def move_game_gui(self, starting_list, game_idx_start_list):
        # model, env, name, folder_name, list_= 'games'
        # TODO fix:
        cur_list = self.ui.games_verticalLayout if starting_list == 'games' else self.ui.ranking_verticalLayout
        removed_widgets = self.remove_game_from_gui(cur_list, game_idx_start_list)
        print(removed_widgets)
        dest_list = 'games' if starting_list == 'rank' else 'games'
        self.add_row(env_used, 'ff', folder_name, 'rank')
        # self.ui.games_verticalLayout.removeItem()
        pass

    def move_game_up_gui(self, game_idx):
        #insertItem(index, a1)
        # self.ui.games_verticalLayout.insertLayout()
        # replaceWidget()
        pass

    def move_game_down_gui(self, game_idx):
        pass

    def get_items_in_list(self, list ='games'):
        if list == 'games':
            games_items = (self.ui.games_verticalLayout.itemAt(i) for i in range(self.ui.games_verticalLayout.count()))
        else:
            games_items = (self.ui.ranking_verticalLayout.itemAt(i) for i in range(self.ui.ranking_verticalLayout.count()))
        return games_items

    def show_traj_imgs(self, games_dir, img_label, timer, count_idx):
        # timer = QTimer(self)
        # TODO loop over imgs in dir until when?
        imgs_nums = [elem.split('game')[1].split('.')[0] for elem in os.listdir(games_dir) if elem.endswith(".png")]
        imgs_nums.sort(key=int)
        imgs = ['game' + str(img_num) + '.png' for img_num in imgs_nums]

        imgs_cycle = cycle(imgs)

        timer.timeout.connect(lambda: self.update_image(os.path.join(games_dir, next(imgs_cycle)), img_label, imgs, timer, count_idx))

        # timer.timeout.connect(lambda: self.check_timer_end(self.counter, imgs, self))
        # timer.start(60 * 1000)
        timer.start(250)
        # self.update_image(os.path.join(games_dir, imgs[self.img_idx]), img_label)
        self.update_image(os.path.join(games_dir, next(imgs_cycle)), img_label, imgs, timer, count_idx)

    # def check_timer_end(self, imgs_cycle, imgs, timer):
    #     print(imgs_cycle, cycle(imgs))
    #     self.counter += 1
    #     if self.counter == imgs[-1]:
    #         print('stop timer')
    #         timer.stop()
    #         timer.deleteLater()

    def update_image(self, image, img_label, imgs, timer, count_idx):
        # img_num = image.split("/")[-1].split('game')[1].split('.')[0]
        # print(img_num, len(imgs)-1)
        # if img_num == len(imgs)-1:
        #     timer.stop()
        pixmap = QPixmap(image)

        if not pixmap.isNull():
            # print('update image')
            img_label.setPixmap(pixmap)
            self.counts[count_idx] += 1
            # self.label.adjustSize()
            # self.resize(pixmap.size())

        if self.counts[count_idx] == len(imgs):
            timer.stop()
            self.counts[count_idx] = 0
            return

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = GamesView(env_used)
#     window.show()
#     sys.exit(app.exec_())
