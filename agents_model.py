from datetime import datetime
import json
import os
import shutil

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, pyqtProperty

from policy_nets.base_policy_net import PolicyNet
from train_policy_net import train_policy
from train_reward_net import train_reward


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
        self.load_from_disk()

    def load_from_disk(self):

        agents_loaded = 0
        envs_loaded = 0

        # for each policy
        for el in os.listdir(self.agents_dir):
            policy_dir = os.path.join(self.agents_dir, el)
            if not os.path.isdir(policy_dir) or el == "__pycache__":
                continue

            # for each environment
            for env in os.listdir(policy_dir):
                env_dir = os.path.join(policy_dir, env)
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
                        with open(trained_policy_info, 'rt') as file:
                            j = json.load(file)
                        j["path"] = trained_policy_dir
                        if "reward_type" not in j or j["reward_type"] == "env" or "reward_key" not in j: # TODO rimuovere la prima parte dell'if
                            continue
                        j["games"] = self.read_agent_games(env, j["reward_key"])
                        self.add_agent(env, trained_policy, j)
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

    def create_agent(self, environment, games):
        # TODO finire di sistemare
        train_reward(environment, games=games)
        policy_net_key = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        train_policy(environment, policy_net_key=policy_net_key)
        self.add_agent(environment, policy_net_key)
        # TODO threads?

    def add_agent(self, environment: str, agent_key: str, agent_value=None) -> bool:

        # TODO gestire caso in cui agent_value=None

        if environment not in self._agents or agent_key in self._agents[environment]:
            return False
        self._agents[environment][agent_key] = agent_value
        self.agent_added.emit(environment, agent_key)
        return True

    def update_agent(self, environment: str, agent_key: str, agent_value) -> bool:
        if environment not in self._agents or agent_key not in self._agents[environment]:
            return False
        self._agents[environment][agent_key] = agent_value
        self.agent_updated.emit(environment, agent_key)
        return True

    def delete_agent(self, environment: str, agent_key: str) -> bool:
        if environment not in self._agents or agent_key not in self._agents[environment]:
            return False
        agent = self._agents[environment].pop(agent_key)
        shutil.rmtree(agent["path"], ignore_errors=True)
        self.agent_deleted.emit(environment, agent_key)
        return True

    def get_agent(self, environment, agent_name):
        return self._agents[environment][agent_name]

    def read_agent_games(self, environment, reward_key):
        trained_reward_info = os.path.join(self.rewards_dir, "conv_reward", environment, reward_key, "training.json")  # TODO cambiare!!!!!! -> rimuovere "conv_reward"
        try:
            with open(trained_reward_info, 'rt') as file:
                j = json.load(file)
            return j["games"]
        except FileNotFoundError:
            print("File not found: " + trained_reward_info)
            return None
