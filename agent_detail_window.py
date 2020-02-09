import os
import time
from threading import Thread
from itertools import cycle
import gym
import gym_minigrid
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMovie, QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QHBoxLayout, QLabel, QPushButton, QWidget, QStyle, QSlider

from agent_detail_ui import Ui_Agent
from games_model import GamesModel
from games_view import GamesListView
from trainer import TrainingManager
# from utils import nparray_to_qpixmap, state_filter

from utils import nparray_to_qpixmap, state_filter, load_net, policies_dir, get_episodes_for_checkpoint, num_max_episodes


class AgentDetailWindow(QMainWindow):

    def __init__(self, agents_model, environment, agent_key):
        super().__init__()

        # save parameters into class fields
        self.agents_model = agents_model
        self.environment = environment
        self.agent_key = agent_key
        self.agent = self.agents_model.get_agent(self.environment, self.agent_key)

        # init UI
        self.ui = Ui_Agent()
        self.ui.setupUi(self)

        # ProgressBar & Slider
        self.ui.progressBarTraining.setFixedWidth(200)
        self.ui.progressBarTraining.setMinimum(0)
        self.ui.sliderTraining.setFixedWidth(200)
        self.ui.sliderTraining.setMaximum(num_max_episodes()/get_episodes_for_checkpoint())
        self.ui.sliderTraining.setVisible(False) if self.agent.running else self.ui.sliderTraining.setVisible(True)
        self.ui.sliderTraining.valueChanged.connect(self.sliderChanged)

        self.setWindowTitle(agent_key)
        self.ui.txt_name.setText(self.agent.name)
        self.games_model = GamesModel(environment, agents_model)
        self.widget_list_games = GamesListView(self.games_model, environment, False, False, False, False)
        self.ui.info_scrollArea.setWidget(self.widget_list_games)
        self.ui.btnPlayPause.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

        # start the thread that plays minigrid with this agent (policy)
        self.game_thread = GameThread(self.agent, environment, self.ui.labelGame)
        self.game_thread.start()

        # update text label and gif label of training status (completed / training)
        self.refresh_training_status()
        self.agents_model.agent_updated.connect(lambda env, ag: self.refresh_training_status() if env==self.environment and ag==self.agent_key else lambda: ...)

        # link slots
        self.ui.btnDeleteAgent.clicked.connect(self.delete)
        self.ui.btnPlayPause.clicked.connect(self.playPauseSlot)

        for game_key in reversed(agents_model.get_agent(environment, agent_key).games):
            self.widget_list_games.add_game(game_key)

    def refresh_training_status(self):
        max_episodes = self.agent.max_episodes
        current_episode = self.agent.episode

        if not self.agent.running or current_episode + 1 == max_episodes: # +1 is used because episode count starts from 0
            self.ui.labelLoading.setText("")
            self.ui.labelLoading.setVisible(False)
            self.ui.labelLoading.setMovie(None)

            if current_episode + 1 == max_episodes:
                self.ui.labelStatus.setText("Status: training completed")
                self.ui.progressBarTraining.setEnabled(False)
                self.ui.btnPlayPause.setVisible(False)
                self.ui.sliderTraining.setVisible(True)
                self.ui.sliderTraining.setValue(num_max_episodes()//get_episodes_for_checkpoint())
            else:
                self.ui.labelStatus.setText("Status: paused")
                self.ui.progressBarTraining.setEnabled(True)
                self.ui.btnPlayPause.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
                self.ui.btnPlayPause.setVisible(True)
                self.ui.sliderTraining.setVisible(True)
                self.ui.sliderTraining.setValue(num_max_episodes()//get_episodes_for_checkpoint())

        else:
            self.ui.labelStatus.setText("Status: training")
            self.ui.labelLoading.setVisible(True)
            self.ui.btnPlayPause.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.ui.btnPlayPause.setVisible(True)
            self.ui.sliderTraining.setVisible(False)
            if self.ui.labelLoading.movie() is None:
                self.gif = QMovie(os.path.join("img", "loading.gif"))
                self.gif.start()
                self.ui.labelLoading.setMovie(self.gif)
            if self.agent != self.game_thread.agent:
                self.game_thread.set_agent(self.agent)

        self.ui.progressBarTraining.setMaximum(max_episodes-1)
        self.ui.progressBarTraining.setValue(current_episode)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.game_thread.interrupt()
        super().closeEvent(a0)

    def delete(self):
        reply = QMessageBox.question(self, "Delete agent", "Are you sure to delete this agent?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            print("deleting: " + self.agents_model.get_agent(self.environment, self.agent_key).folder)
            self.agents_model.delete_agent(self.environment, self.agent_key)
            print("successfully deleted")
            self.close()

    def playPauseSlot(self):

        if self.agents_model.is_agent_training(self.environment, self.agent_key):
            if not self.agent.running:
                self.agents_model.play_agent(self.environment, self.agent_key)
                self.ui.sliderTraining.setVisible(False)
            else:
                self.agents_model.pause_agent(self.environment, self.agent_key)
                self.ui.sliderTraining.setVisible(True)
        else:
            self.agents_model.resume_agent_training(self.environment, self.agent_key)

        # if self.agent.running:
        #     self.agent.pause()
        #     self.ui.SliderTraining.setVisible(True)
        # else:
        #     self.agent.play()
        #     self.ui.SliderTraining.setVisible(False)

        self.refresh_training_status()

    def sliderChanged(self):
        episodes_for_checkpoint = get_episodes_for_checkpoint()

        s_val = self.ui.sliderTraining.value()
        # self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        policy_name = policies_dir()
        policies = [int(el.split('-')[1].split('.')[0]) for el in os.listdir(os.path.join(policy_name, self.environment, self.agent_key)) if el.endswith('.pth')]

        # new value is bigger than max value
        if s_val > int(max(policies) / episodes_for_checkpoint):
            s_val = int(max(policies) / episodes_for_checkpoint)
            self.ui.sliderTraining.setValue(s_val)

        policy_net = os.path.join(policy_name, self.environment, self.agent_key, 'policy_net-' + str(s_val * episodes_for_checkpoint) + '.pth')
        new_agent = load_net(policy_net)
        if new_agent.episode != self.game_thread.agent:
            self.game_thread.set_agent(new_agent)


class GameThread(Thread):
    def __init__(self, agent, environment, game_qlabel):
        Thread.__init__(self)
        self.game_widget = game_qlabel
        self.env = gym.make(environment)
        self.agent = agent
        self._running = True

    def run(self):
        state = self.env.reset()
        done = False
        while self._running:
            img = self.env.render("pixmap")
            try:
                self.game_widget.setPixmap(nparray_to_qpixmap(img))
            except RuntimeError:
                break

            time.sleep(0.1)

            if not self._running:
                break

            if done:
                state = self.env.reset()
                done = False
                continue

            action = self.agent.sample_action(state_filter(state))
            state, reward, done, info = self.env.step(action)
            state = self.env.gen_obs()

    def interrupt(self):
        self._running = False

    def set_agent(self, new_agent):
        self.agent = new_agent
        self.env.reset()
