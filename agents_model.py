import json
import os
import shutil

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, pyqtProperty

from policy_nets.base_policy_net import PolicyNet


class AgentsModel(QObject):

    environment_added = pyqtSignal(str)
    environment_deleted = pyqtSignal(str)

    agent_added = pyqtSignal(str, str)  # TODO cambiare
    agent_updated = pyqtSignal(str, str)
    agent_deleted = pyqtSignal(str, str)  # TODO cambiare

    def __init__(self):
        super().__init__(parent=None)
        self._agents = {}
        self.load_from_disk()

    def load_from_disk(self):

        agents_loaded = 0
        envs_loaded = 0

        agents_dir = "policy_nets"
        # for each policy
        for el in os.listdir(agents_dir):
            policy_dir = os.path.join(agents_dir, el)
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
                    #self.agents[env] = {}
                    #self.agents[env] = []
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
                        self.add_agent(env, trained_policy, j)
                        # self.agents[env][trained_policy]
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

    def add_agent(self, environment: str, agent_key: str, agent_value) -> bool:
        added = self.update_agent(environment, agent_key, agent_value)
        if added:
            self.agent_added.emit(environment, agent_key)
        return added

    def update_agent(self, environment: str, agent_key: str, agent_value) -> bool:
        if environment not in self._agents or agent_key in self._agents[environment]: # TODO change
            return False
        self._agents[environment][agent_key] = agent_value
        self.agent_updated.emit(environment, agent_key)
        return True

    def delete_agent(self, environment: str, agent_key: str) -> bool:
        if environment not in self._agents or agent_key not in self._agents[environment]:
            return False
        agent = self._agents[environment].pop(agent_key)
        shutil.rmtree(agent["path"], ignore_errors=True)
        return True

    def get_agent(self, environment, agent_name):
        return self._agents[environment][agent_name]

