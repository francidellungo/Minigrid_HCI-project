import os
import time
from threading import Thread

from itertools import cycle
import gym
import gym_minigrid
from PyQt5 import QtGui
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QMovie, QPixmap
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QHBoxLayout, QLabel, QPushButton, QWidget

from agent_detail_ui import Ui_Agent
from utils import load_last_policy, state_filter, get_last_policy_episode

games_path = 'games'


class AgentDetailWindow(QMainWindow):

    def __init__(self, agents_model, environment, agent_key):
        super().__init__()

        # save parameters into class fields
        self.agents_model = agents_model
        self.environment = environment
        self.agent_key = agent_key

        # init UI
        self.ui = Ui_Agent()
        self.ui.setupUi(self)
        self.setWindowTitle(agent_key)

        # update text label and gif label of training status (completed / training)
        self.refresh_training_status()
        self.agents_model.agent_updated.connect(lambda env, ag: self.refresh_training_status() if env==self.environment and ag==self.agent_key else lambda: ...)

        # link slot to delete the agent
        self.ui.btnDeleteAgent.clicked.connect(self.delete)

        # display on the right side of the window the games used to train the reward used to train this policy
        self.display_games()

        # start the thread that plays minigrid with this agent (policy)
        self.game_thread = GameThread(agents_model, environment, agent_key, self.ui.labelGame)
        self.game_thread.start()

    def refresh_training_status(self):
        agent = self.agents_model.get_agent(self.environment, self.agent_key)
        max_episodes = agent["max_episodes"]
        current_episode = get_last_policy_episode(agent["path"]) or 0

        if current_episode + 1 == max_episodes: # +1 is used because episode count starts from 0
            self.ui.labelStatus.setText("Status: training completed")
            self.ui.progressBarTraining.setEnabled(False)
            self.ui.labelLoading.setText("")
        else:
            self.ui.labelStatus.setText("Status: training")
            self.gif = QMovie(os.path.join("img", "loading.gif"))
            self.gif.start()
            self.ui.labelLoading.setMovie(self.gif)

        self.ui.progressBarTraining.setMaximum(max_episodes)
        self.ui.progressBarTraining.setValue(current_episode+1)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.game_thread.interrupt()
        super().closeEvent(a0)

    def delete(self):
        reply = QMessageBox.question(self, "Delete agent", "Are you sure to delete this agent?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            print("deleting: " + self.agents_model.get_agent(self.environment, self.agent_key)["path"])
            self.agents_model.delete_agent(self.environment, self.agent_key)
            print("successfully deleted")
            self.close()

    def display_games(self):
        games = self.agents_model.get_agent(self.environment, self.agent_key)["games"]
        for game in games:
            self.add_row(game)

    def add_row(self, folder_name):
        horiz = QHBoxLayout()
        horiz.addWidget(QLabel(folder_name))
        # pixmap = QPixmap(path_of_image + '0' + '.png')
        pixmap = QPixmap('games/MiniGrid-Empty-6x6-v0/' + folder_name + '/game0.png')
        # print(path_of_image + '0' + '.png')
        img_label = QLabel()
        img_label.setPixmap(pixmap)
        horiz.addWidget(img_label)
        horiz.addWidget(QPushButton('info'))
        timer = QTimer()

        dir_path = os.path.join(games_path, self.environment, folder_name)

        img_label.enterEvent = lambda ev: self.show_traj_imgs(dir_path, img_label, timer)
        img_label.leaveEvent = lambda ev: self.stop_show_traj(dir_path, img_label, timer)

        self.ui.info_verticalLayout_2.addLayout(horiz)

        # self.update_image(0)
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

class GameThread(Thread):
    def __init__(self, agents_model, environment, agent_name, game_qlabel):
        Thread.__init__(self)
        self.game_widget = game_qlabel
        self.env = gym.make(environment)
        self.agent = load_last_policy(agents_model.get_agent(environment, agent_name)["path"])
        self._running = True

    def run(self):
        state = self.env.reset()
        done = False
        while self._running:
            img = self.env.render("pixmap")
            try:
                self.game_widget.setPixmap(img)
            except RuntimeError:
                break

            time.sleep(0.1)

            if not self._running:
                break

            if done:
                state = self.env.reset()
                done = False
                continue

            action = self.agent.sample_action(state_filter(self.agent, state))
            state, reward, done, info = self.env.step(action)

    def interrupt(self):
        self._running = False
