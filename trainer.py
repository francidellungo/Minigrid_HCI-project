from datetime import datetime
from threading import Thread

from train_policy_net import train_policy
from train_reward_net import train_reward


class TrainingManager:
    threads = []

    @staticmethod
    def train_new_agent(environment, games, agents_model, on_episode_end):
        callbacks = [{
            "on_episode_end": on_episode_end,
            "on_train_end": lambda key: TrainingManager.threads.remove(th) and agents_model.add_agent(environment, key)
        }]
        th = TrainerThread(environment, games, callbacks)
        TrainingManager.threads.append(th)
        th.start()


    # TODO implement
    # @staticmethod
    # def stop_all_trainings():
    #     for thread in TrainingManager.threads:
    #         thread.


class TrainerThread(Thread):

    def __init__(self, environment, games, callbacks=[]):
        super().__init__()
        self.environment = environment
        self.games = games
        self.callbacks = callbacks

    def run(self):
        reward_net_path = train_reward(self.environment, games=self.games)
        policy_net_key = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        train_policy(self.environment, reward_net_path=reward_net_path, policy_net_key=policy_net_key, callbacks=self.callbacks)
