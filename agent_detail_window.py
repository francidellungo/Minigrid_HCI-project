import os
import random
import sys
import time
from threading import Thread

import gym
import gym_minigrid
from PyQt5 import QtGui
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QMainWindow, QApplication, QHBoxLayout, QPushButton, QLabel, QWidget, QMessageBox

from agent_detail_ui import Ui_Agent
from agents_model import AgentsModel
from agents_ui import Ui_Agents
from utils import load_last_policy, state_filter


class AgentDetailWindow(QMainWindow):

    def __init__(self, agents_model, environment, agent_name):
        super().__init__()

        self.agents_model = agents_model
        self.environment = environment
        self.agent_name = agent_name

        self.ui = Ui_Agent()
        self.ui.setupUi(self)
        self.setWindowTitle(agent_name)

        r = 50 * random.randint(0, 2) # TODO sistemare r
        if r == 100:
            self.ui.labelStatus.setText("Status: training completed")
            self.ui.progressBarTraining.setEnabled(False)
            self.ui.labelLoading.setText("")
        else:
            self.ui.labelStatus.setText("Status: training")
            self.gif = QMovie(os.path.join("img", "loading.gif"))
            self.gif.start()
            self.ui.labelLoading.setMovie(self.gif)
        self.ui.progressBarTraining.setValue(r)

        self.ui.btnDeleteAgent.clicked.connect(self.delete)

        self.game_thread = GameThread(agents_model, environment, agent_name, self.ui.labelGame)
        self.game_thread.start()

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        super().closeEvent(a0)
        self.game_thread.interrupt()

    def delete(self):
        reply = QMessageBox.question(self, "Delete agent", "Are you sure to delete this agent?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            print("deleting: " + self.agents_model[self.environment][self.agent_name]["path"])
            self.agents_model.delete_agent(self.environment, self.agent_name)


class GameThread(Thread):
    def __init__(self, agents_model, environment, agent_name, game_qlabel):
        Thread.__init__(self)
        self.game_widget = game_qlabel
        self.env = gym.make(environment)
        self.agent = load_last_policy(agents_model[environment][agent_name]["path"])
        self._running = True

    def run(self):
        state = self.env.reset()
        done = False
        while self._running:
            img = self.env.render("pixmap")
            self.game_widget.setPixmap(img)
            time.sleep(0.1)
            if done:
                state = self.env.reset()
                done = False
                continue

            action = self.agent.sample_action(state_filter(self.agent, state))
            state, reward, done, info = self.env.step(action)

    def interrupt(self):
        self._running = False
