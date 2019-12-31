import os
import time
from threading import Thread

import gym
import gym_minigrid
from PyQt5 import QtGui
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QMainWindow, QMessageBox

from agent_detail_ui import Ui_Agent
from utils import load_last_policy, state_filter, get_last_policy_episode


class AgentDetailWindow(QMainWindow):

    def __init__(self, agents_model, environment, agent_name):
        super().__init__()

        self.agents_model = agents_model
        self.environment = environment
        self.agent_name = agent_name

        self.ui = Ui_Agent()
        self.ui.setupUi(self)
        self.setWindowTitle(agent_name)

        self.refresh_training_status()

        self.ui.btnDeleteAgent.clicked.connect(self.delete)

        self.game_thread = GameThread(agents_model, environment, agent_name, self.ui.labelGame)
        self.game_thread.start()

    def refresh_training_status(self):
        agent = self.agents_model.get_agent(self.environment, self.agent_name)
        max_episodes = agent["max_episodes"]
        current_episode = get_last_policy_episode(agent["path"])

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
            print("deleting: " + self.agents_model.agents[self.environment][self.agent_name]["path"])
            self.agents_model.delete_agent(self.environment, self.agent_name)
            print("successfully deleted")
            self.close()


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
