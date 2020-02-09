import importlib
import pickle
from datetime import datetime
import json
import os
import shutil
from threading import Semaphore

import torch
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, pyqtProperty

from policy_nets.base_policy_net import PolicyNet
from train_policy_net import train_policy
from train_reward_net import train_reward
from trainer import TrainingManager
from utils import *


class AgentsModel(QObject):

    environment_added = pyqtSignal(str)
    environment_deleted = pyqtSignal(str)

    agent_added = pyqtSignal(str, str)  # TODO cambiare
    agent_updated = pyqtSignal(str, str)
    agent_deleted = pyqtSignal(str, str)  # TODO cambiare

    def __init__(self):
        super().__init__(parent=None)
        self._agents = {}
        self.agents_dir = policies_dir()
        self.rewards_dir = rewards_dir()
        self.games_dir = games_dir()
        self.device = auto_device()
        self.locks = {}
        self.load_from_disk()

    def load_from_disk(self):

        agents_loaded = 0
        envs_loaded = 0

        # for each environment in agents_dir
        for env in os.listdir(self.agents_dir):
            env_dir = os.path.join(self.agents_dir, env)
            if not os.path.isdir(env_dir) or env == "__pycache__":
                continue

            # env is a name (string) of an environment
            if env not in self._agents:
                self.add_environment(env)
                envs_loaded += 1

            # for each trained policy (of this type and in this environment)
            for trained_policy in os.listdir(env_dir):
                trained_policy_dir = os.path.join(env_dir, trained_policy)
                if not os.path.isdir(trained_policy_dir):
                    continue

                trained_policy_info = os.path.join(trained_policy_dir, "training.json")
                try:
                    self.add_agent(env, trained_policy)
                    agents_loaded += 1
                except FileNotFoundError:
                    print("File not found: " + trained_policy_info)

        print("loaded {} agents from {} environments".format(agents_loaded, envs_loaded))

        if os.path.exists(self.games_dir):
            # for each environment in games_dir
            for env in os.listdir(self.games_dir):
                env_dir = os.path.join(self.games_dir, env)
                if not os.path.isdir(env_dir):
                    continue
                self.add_environment(env)

    def add_environment(self, environment: str) -> bool:
        if environment in self._agents:
            return False
        self._agents[environment] = {}
        self.environment_added.emit(environment)
        return True

    def delete_environment(self, environment: str) -> bool:
        if environment not in self._agents:
            return False
        self._agents.pop(environment)
        shutil.rmtree(os.path.join(self.agents_dir, environment), ignore_errors=True)
        shutil.rmtree(os.path.join(self.rewards_dir, environment), ignore_errors=True)
        shutil.rmtree(os.path.join(self.games_dir, environment), ignore_errors=True)
        self.environment_deleted.emit(environment)
        return True

    def get_environments(self):
        return self._agents.keys()

    def create_agent(self, environment: str, games: list):
        TrainingManager.train_new_agent(environment, games, self, lambda agent: self.agent_updated.emit(environment, agent.key))

    def add_agent(self, environment: str, agent_key: str, agent:PolicyNet=None) -> bool:
        if environment not in self._agents:
            self.add_environment(environment)

        if agent_key in self._agents[environment]:
            return False

        if agent is None:
            agent = self.load_agent(environment, agent_key)

        self._agents[environment][agent_key] = agent
        self.agent_added.emit(environment, agent_key)

        self.locks[agent] = Semaphore(1)

        return True

    def delete_agent(self, environment: str, agent_key: str) -> bool:
        if environment not in self._agents or agent_key not in self._agents[environment]:
            return False
        if TrainingManager.is_agent_training(environment, agent_key):
            TrainingManager.interrupt_training(self, environment, agent_key)

        agent = self._agents[environment].pop(agent_key)
        shutil.rmtree(agent.folder, ignore_errors=True)
        self._delete_agent_games(environment, agent)
        self.agent_deleted.emit(environment, agent_key)
        return True

    def _delete_agent_games(self, environment, agent):
        games = agent.games
        for game in games:
            with open(os.path.join("games", environment, game, "game.json"), 'rt') as file:
                j = json.load(file)
            if not j["to_delete"]:
                continue
            if self.game_used_by_some_agent(environment, game):
                continue
            removing_dir = os.path.join("games", environment, game)
            shutil.rmtree(removing_dir, ignore_errors=True)

    def game_used_by_some_agent(self, environment, game):
        for agent_key in self._agents[environment]:
            agent = self._agents[environment][agent_key]
            if game in agent.games:
                return True
        return False

    def get_agent(self, environment: str, agent_key: str):
        try:
            return self._agents[environment][agent_key]
        except KeyError:
            return None

    def read_agent_games(self, environment, reward_key):
        trained_reward_info = os.path.join(self.rewards_dir, environment, reward_key, "training.json")
        try:
            with open(trained_reward_info, 'rt') as file:
                j = json.load(file)
            return j["games"]
        except FileNotFoundError:
            print("File not found: " + trained_reward_info)
            return None

    def load_agent(self, environment: str, agent_key: str, num: int = None):
        agent_dir = os.path.join(self.agents_dir, environment, agent_key)

        # module_path, _ = policy_net_file.rsplit(".", 1)
        # net_module = importlib.import_module(".".join(os.path.split(module_path)))
        # reward_net = net_module.get_net(get_input_shape(), get_num_actions(), environment, agent_key, folder=agent_dir).to(self.device)
        if num is None:
            agent = pickle.load(open(os.path.join(agent_dir, "net.pkl"), "rb")).to(self.device).load_last_checkpoint()
        else:
            agent = pickle.load(open(os.path.join(agent_dir, "net.pkl"), "rb")).to(self.device).load_checkpoint(num)
        return agent

    # TODO change
    def load_agent_value(self, environment: str, agent_key: str):
        trained_agent_dir = os.path.join(self.agents_dir, environment, agent_key)
        trained_agent_info = os.path.join(trained_agent_dir, "training.json")
        try:
            with open(trained_agent_info, 'rt') as file:
                j = json.load(file)
            j["path"] = trained_agent_dir
            try:
                j["games"] = self.read_agent_games(environment, j["reward_net_key"])
            except KeyError:
                j["games"] = []
            return j
        except FileNotFoundError:
            print("File not found: " + trained_agent_info)
            return None

    def get_agents(self, environment: str):
        if environment in self._agents:
            return self._agents[environment].keys()
        return []

    def is_agent_training(self, environment, agent_key):
        return TrainingManager.is_agent_training(environment, agent_key)

    def resume_agent_training(self, environment, agent_key):
        TrainingManager.resume_agent_training(environment, self, self._agents[environment][agent_key], lambda agent: self.agent_updated.emit(environment, agent_key))

    def play_agent(self, environment, agent_key):
        self.locks[self._agents[environment][agent_key]].release()
        self.get_agent(environment, agent_key).play()
        self.agent_updated.emit(environment, agent_key)

    def pause_agent(self, environment, agent_key):
        self.locks[self._agents[environment][agent_key]].acquire()
        self.get_agent(environment, agent_key).pause()
        self.agent_updated.emit(environment, agent_key)

    def get_agent_lock(self, environment, agent_key):
        agent = self.get_agent(environment, agent_key)
        if agent is None:
            return Semaphore(1)
        return self.locks[agent]
