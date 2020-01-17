from datetime import datetime
from threading import Thread, Semaphore

from train_policy_net import train_policy
from train_reward_net import train_reward


class TrainingManager:
    threads = []

    @staticmethod
    def train_new_agent(environment, games, agents_model, on_episode_end):
        policy_callbacks = [{
            "on_train_begin": lambda agent: agents_model.add_agent(environment, agent.key, agent),
            "on_episode_end": on_episode_end,
            "on_train_end": lambda agent: TrainingManager.threads.remove(th)
        }]
        th = TrainerThread(environment, games, policy_callbacks)
        TrainingManager.threads.append(th)
        th.start()


    # TODO implement
    # @staticmethod
    # def stop_all_trainings():
    #     for thread in TrainingManager.threads:
    #         thread.


class TrainerThread(Thread):

    def __init__(self, environment, games, policy_callbacks=[]):
        super().__init__()
        self.environment = environment
        self.games = games
        self.policy_callbacks = policy_callbacks
        self.reward_train_ended_lock = Semaphore(0)
        self.policy_callbacks.append({"on_train_begin": lambda agent: self.reward_train_ended_lock.acquire()})

    def run(self):
        reward_trainer = RewardTrainerThread(self.environment, self.games, self.reward_train_ended_lock)
        reward_trainer.start()
        reward_net_path = reward_trainer.get_reward_net_folder()
        policy_net_key = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        train_policy(self.environment, reward_net_path=reward_net_path, policy_net_key=policy_net_key, callbacks=self.policy_callbacks)


class RewardTrainerThread(Thread):

    def __init__(self, environment, games, reward_train_ended_lock=None):
        super().__init__()
        self.environment = environment
        self.games = games
        self.folder_lock = Semaphore(0)
        self.reward_folder = None
        self.callbacks = [{"on_train_begin": lambda reward_net: self.set_reward_net_folder(reward_net.folder)}]
        self.reward_train_ended_lock = reward_train_ended_lock
        if reward_train_ended_lock is not None:
            self.callbacks.append({"on_train_end": lambda reward_net: self.reward_train_ended_lock.release()})

    def run(self):
        train_reward(self.environment, games=self.games, callbacks=self.callbacks)

    def set_reward_net_folder(self, folder):
        self.reward_folder = folder
        self.folder_lock.release()

    def get_reward_net_folder(self):
        self.folder_lock.acquire()
        return self.reward_folder
