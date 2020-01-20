import importlib
import pickle
from datetime import datetime
import json
import os
import shutil

import torch
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, pyqtProperty

from policy_nets.base_policy_net import PolicyNet
from train_policy_net import train_policy
from train_reward_net import train_reward
from trainer import TrainingManager
from utils import get_input_shape, get_num_actions


class AgentsModel(QObject):

    environment_added = pyqtSignal(str)
    environment_deleted = pyqtSignal(str)

    agent_added = pyqtSignal(str, str)  # TODO cambiare
    agent_updated = pyqtSignal(str, str)
    agent_deleted = pyqtSignal(str, str)  # TODO cambiare

    def __init__(self):
        super().__init__(parent=None)
        self._agents = {}
        self.agents_dir = "policy_nets"
        self.rewards_dir = "reward_nets"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_from_disk()

    def load_from_disk(self):

        agents_loaded = 0
        envs_loaded = 0

        # for each environment
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
                    # with open(trained_policy_info, 'rt') as file:
                    #     j = json.load(file)
                    # j["path"] = trained_policy_dir
                    # if "reward_type" not in j or j["reward_type"] == "env" or "reward_net_key" not in j: # TODO rimuovere la prima parte dell'if
                    #     continue
                    # j["games"] = self.read_agent_games(env, j["reward_net_key"])
                    self.add_agent(env, trained_policy)
                    agents_loaded += 1
                except FileNotFoundError:
                    print("File not found: " + trained_policy_info)

        print("loaded {} agents from {} environments".format(agents_loaded, envs_loaded))

    def add_environment(self, environment: str) -> bool:
        if environment in self._agents:
            return False
        self._agents[environment] = {}
        self.environment_added.emit(environment)
        return True

    def pop_environment(self, environment: str):
        if environment not in self._agents:
            return False
        env = self._agents.pop(environment)
        self.environment_deleted.emit(environment)
        return env

    def delete_environment(self, environment: str) -> bool:
        if environment not in self._agents:
            return False
        self.pop_environment(environment)
        return True

    def get_environments(self):
        return self._agents.keys()

    def create_agent(self, environment: str, games: list):
        TrainingManager.train_new_agent(environment, games, self, lambda agent: self.add_agent(environment, agent.key, agent) or self.update_agent(environment, agent.key))

    def add_agent(self, environment: str, agent_key: str, agent:PolicyNet=None) -> bool:
        if environment not in self._agents or agent_key in self._agents[environment]:
            return False

        if agent is None:
            agent = self.load_agent(environment, agent_key)

        self._agents[environment][agent_key] = agent
        self.agent_added.emit(environment, agent_key)
        return True

    def update_agent(self, environment: str, agent_key: str) -> bool:
        if environment not in self._agents or agent_key not in self._agents[environment]:
            return False
        self.agent_updated.emit(environment, agent_key)
        return True

    def delete_agent(self, environment: str, agent_key: str) -> bool:
        if environment not in self._agents or agent_key not in self._agents[environment]:
            return False
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
        return self._agents[environment][agent_key]

    def read_agent_games(self, environment, reward_key):
        trained_reward_info = os.path.join(self.rewards_dir, "conv_reward", environment, reward_key, "training.json")  # TODO cambiare!!!!!! -> rimuovere "conv_reward"
        try:
            with open(trained_reward_info, 'rt') as file:
                j = json.load(file)
            return j["games"]
        except FileNotFoundError:
            print("File not found: " + trained_reward_info)
            return None

    def load_agent(self, environment: str, agent_key: str):
        agent_dir = os.path.join(self.agents_dir, environment, agent_key)

        # module_path, _ = policy_net_file.rsplit(".", 1)
        # net_module = importlib.import_module(".".join(os.path.split(module_path)))
        # reward_net = net_module.get_net(get_input_shape(), get_num_actions(), environment, agent_key, folder=agent_dir).to(self.device)
        agent = pickle.load(open(os.path.join(agent_dir, "net.pkl"), "rb")).to(self.device).load_last_checkpoint()
        return agent

    def load_agent_value(self, environment: str, agent_key: str):
        trained_agent_dir = os.path.join(self.agents_dir, environment, agent_key)
        trained_agent_info = os.path.join(trained_agent_dir, "training.json")
        try:
            with open(trained_agent_info, 'rt') as file:
                j = json.load(file)
            j["path"] = trained_agent_dir
            j["games"] = self.read_agent_games(environment, j["reward_net_key"])
            return j
        except FileNotFoundError:
            print("File not found: " + trained_agent_info)
            return None

    def get_agents(self, environment: str):
        if environment in self._agents:
            return self._agents[environment].keys()
        return []
