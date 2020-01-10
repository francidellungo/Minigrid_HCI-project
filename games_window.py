from datetime import datetime
import sys
import gym
import os

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, pyqtSignal
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
        # self.new_game_view_Dialog = NewGameView(env)

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
        self.view.ui.new_game_pb.clicked.connect(lambda: self.create_new_game(env))
        # self.view.ui.remove_game_pb.clicked.connect(lambda: self.model.remove_game('games', 0))
        # self.view.ui.remove_game_pb.clicked.connect(lambda: self.games_model.remove_game('games', 0))
        self.view.ui.train_pb.clicked.connect(self.train_agent_slot)

    def create_new_game(self, env):
        """
        creation of a new game
        :param env: current environment used
        :param name: name for the new game (really needed?)
        :return:
        """
        # TODO play minigrid to be fixed
        # game_dir = open_newGame_dialog(env)
        new_game_dialog = NewGameView(env)
        new_game_dialog.play_new_game()

        if new_game_dialog.accept:
            print('new game saved:', new_game_dialog.game_folder)
            self.model.new_game(env, new_game_dialog.game_folder, 'game ' + str(self.model.n_games))
        else:
            print('not saved')

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
        self.env_name = environment
        self.keyDownCb = None
        self.done = False
        self.game_folder = None
        self.ui.game_buttonBox.button(QtWidgets.QDialogButtonBox.Save).setEnabled(False)

    def accept(self) -> str:
        print('accept event', self.game_folder)
        self.close()
        return self.game_folder

    def setKeyDownCb(self, callback):
        self.keyDownCb = callback

    def keyPressEvent(self, e):
        print('press ', self.keyDownCb is None, e.key())
        if self.keyDownCb is None:
            return

        keyName = None
        if e.key() == Qt.Key_4:
            keyName = 'LEFT'
            print('left')
        elif e.key() == Qt.Key_6:
            keyName = 'RIGHT'
            print('right')
        elif e.key() == Qt.Key_8:
            keyName = 'UP'
        elif e.key() == Qt.Key_2:
            keyName = 'DOWN'
        # elif e.key() == Qt.Key_Space:
        #     keyName = 'SPACE'
        # elif e.key() == Qt.Key_Return:
        #     keyName = 'RETURN'
        # elif e.key() == Qt.Key_Alt:
        #     keyName = 'ALT'
        # elif e.key() == Qt.Key_Control:
        #     keyName = 'CTRL'
        # elif e.key() == Qt.Key_PageUp:
        #     keyName = 'PAGE_UP'
        # elif e.key() == Qt.Key_PageDown:
        #     keyName = 'PAGE_DOWN'
        # elif e.key() == Qt.Key_Backspace:
        #     keyName = 'BACKSPACE'
        elif e.key() == Qt.Key_Escape:
            keyName = 'ESCAPE'
        print(' keyName : ', keyName)

        if keyName == None:
            return
        self.keyDownCb(keyName)

    def keyDownCb_f(self, keyName):
        # if keyName == 'ESCAPE':
        #     sys.exit(0)

        # if keyName == 'BACKSPACE':
        #     reset_env(self.env)
        #     return

        action = 0

        if keyName == 'LEFT':
            action = self.env.actions.left
        elif keyName == 'RIGHT':
            action = self.env.actions.right
        elif keyName == 'UP':
            action = self.env.actions.forward

        elif keyName == 'SPACE':
            action = self.env.actions.toggle
        elif keyName == 'PAGE_UP':
            action = self.env.actions.pickup
        elif keyName == 'PAGE_DOWN':
            action = self.env.actions.drop

        # elif keyName == 'RETURN':
        #     action = env.actions.done
        #     #action = 'exit_game'

        # Screenshot functionality
        # elif keyName == 'ALT':
        #     screen_path = options.env_name + '.png'
        #     print('saving screenshot "{}"'.format(screen_path))
        #     pixmap = env.render('pixmap')
        #     pixmap.save(screen_path)
        #     return

        else:
            print("unknown key %s" % keyName)
            return

        # Update state
        # act_action(env, action)
        self.act_action(self.env, action)

    def act_action(self, env, action):
        """
        calculate new state (obs), save image of the state and if finished reset the environment
        :param env: gym environment used
        :param action: action taken
        :return:
        """
        global game_directory
        # if action == env.actions.done:
        #     done = True
        # else:
        obs, reward, done, info = env.step(action)
        print("state: ", self.state_filter(obs))

        # Save state
        game_info['trajectory'].append(self.state_filter(obs).tolist())
        game_info['rewards'].append(reward)

        print('step=%s, reward=%.2f' % (env.step_count, reward))

        # Save screenshots
        screenshot_file = 'game' + str(env.step_count) + '.png'
        pixmap = env.render('pixmap')
        screenshots.append((screenshot_file, pixmap))

        # self.ui.game_label.setPixmap(pixmap)

        if done:
            # Save images and json

            # Create new folder to save images and json
            k = 1
            original_game_directory = game_directory
            while os.path.exists(game_directory):
                game_directory = original_game_directory + "_" + str(k)
                k += 1
            os.makedirs(game_directory)

            # Save image of each state
            for screenshot_file, pixmap in screenshots:
                pixmap.save(os.path.join(game_directory, screenshot_file))

            game_info["score"] = sum(game_info["rewards"])
            with open(os.path.join(game_directory, 'game.json'), 'w+') as game_file:
                json.dump(game_info, game_file, ensure_ascii=False)

            print('done!', len(game_info['trajectory']))

            # sys.exit(0)
            # TODO change
            # self.close()
            self.done = True

        if action == env.actions.done:
            return obs, None, True, None
        return obs, reward, done, info

    def play_new_game(self):
        game_label = self.ui.game_label
        self.reset_env(self.env, self.env_name)
        pixmap = self.env.render('pixmap')
        print('pixmap')
        # print(type(self.env.render('human')), type(self.env.render('pixmap')))
        game_label.setPixmap(pixmap)
        self.show()
        self.setKeyDownCb(self.keyDownCb_f)

        self.done = False
        while not self.done:
            pixmap = self.env.render('pixmap')
            game_label.setPixmap(pixmap)

            if self.done is True:
                pixmap = self.env.render('pixmap')
                game_label.setPixmap(pixmap)
                self.ui.game_buttonBox.button(QtWidgets.QDialogButtonBox.Save).setEnabled(True)
                break

        # return self.game_folder

    def state_filter(self, state):
        return state['image'][:, :, 0]

    def reset_env(self, env, env_name):
        """
        reset the environment, initialize game_name, game_info and directory

        :param env: gym environment used
        :return:
        """
        global game_name, game_info, game_directory, screenshots
        state = env.reset()
        env.render()
        # if hasattr(env, 'mission'):
        #     print('Mission: %s' % self.env.mission)

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

        game_directory = os.path.join(games_path, env_name, str(game_name))

        screenshot_file = 'game' + str(env.step_count) + '.png'
        pixmap = env.render('pixmap')
        screenshots = [(screenshot_file, pixmap)]

        self.game_folder = game_name

        return state, game_directory, pixmap

    # def accept(self) -> None:
    #     self.close()
    #     return game_directory

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GamesController(env_used)
    # window.show()
    sys.exit(app.exec_())
