import os
from datetime import datetime
from threading import Thread, Semaphore

from train_policy_net import train_policy
from train_reward_net import train_reward
from utils import policies_dir, load_net, rewards_dir


class TrainingManager:
    threads = []

    @staticmethod
    def train_new_agent(environment, games, agents_model, on_episode_end):
        policy_callbacks = [{
            "on_train_begin": lambda agent: agents_model.add_agent(environment, agent.key, agent),
            "before_update": lambda agent: (agents_model.get_agent_lock(environment, agent.key).acquire(), agents_model.get_agent_lock(environment, agent.key).release()),
            "on_episode_end": on_episode_end,
            "on_train_end": lambda agent: th in TrainingManager.threads and TrainingManager.threads.remove(th)
        }] + [{"on_episode_end": lambda agent: (agents_model.get_agent_lock(environment, agent.key).acquire(), agents_model.get_agent_lock(environment, agent.key).release())}]
        th = PolicyTrainerThread(environment, games, policy_callbacks, train_reward_also=True)
        TrainingManager.threads.append(th)
        th.start()

    @staticmethod
    def resume_agent_training(environment, agents_model, agent, on_episode_end):
        policy_callbacks = [{
            "on_train_begin": lambda agent: agents_model.agent_updated.emit(environment, agent.key),
            "before_update": lambda agent: (agents_model.get_agent_lock(environment, agent.key).acquire(), agents_model.get_agent_lock(environment, agent.key).release()),
            "on_episode_end": on_episode_end,
            "on_train_end": lambda agent: TrainingManager.threads.remove(th)
        }] + [{"on_episode_end": lambda agent: (agents_model.get_agent_lock(environment, agent.key).acquire(), agents_model.get_agent_lock(environment, agent.key).release())}]
        th = PolicyTrainerThread(environment, policy_callbacks=policy_callbacks, train_reward_also=False, policy=agent, policy_net_key=agent.key)
        TrainingManager.threads.append(th)
        th.start()

    @staticmethod
    def is_agent_training(environment, agent_key):
        return any([th.environment == environment and th.policy_net_key == agent_key for th in TrainingManager.threads])

    @staticmethod
    def interrupt_training(agents_model, environment, agent_key):
        toRemove = -1
        for i, th in enumerate(TrainingManager.threads):
            if th.environment == environment and th.policy_net_key == agent_key:
                toRemove = i
                break

        if toRemove != -1:
            TrainingManager.threads[toRemove].interrupt()
            agents_model.get_agent_lock(environment, agent_key).release()
            TrainingManager.threads.pop(toRemove)

    @staticmethod
    def interrupt_all_trainings():
        for thread in TrainingManager.threads:
            thread.interrupt()
        TrainingManager.threads = []


class PolicyTrainerThread(Thread):

    def __init__(self, environment, games=None, policy_callbacks=[], train_reward_also=True, policy_net_key=None, policy=None):
        super().__init__()
        self.environment = environment
        self.games = games
        self.policy_callbacks = policy_callbacks
        self.running = True
        self.policy_callbacks.append({"on_episode_end": lambda agent: self.running or agent.interrupt()})
        self.train_reward_also = train_reward_also
        self.policy_net_key = policy_net_key
        self.policy = policy
        if train_reward_also:
            self.reward_train_ended_lock = Semaphore(0)
            self.policy_callbacks.append({"on_train_begin": lambda agent: (self.set_policy(agent), self.reward_train_ended_lock.acquire())})
            self.reward_trainer = RewardTrainerThread(self.environment, self.games, self.reward_train_ended_lock)

    def run(self):
        if self.train_reward_also:
            self.reward_trainer.start()
            reward_net_path = self.reward_trainer.get_reward_net_folder()
            if self.policy_net_key is None:
                self.policy_net_key = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            train_policy(self.environment, reward_net_arg=reward_net_path, policy_net_key=self.policy_net_key, callbacks=self.policy_callbacks)
        else:
            episodes = self.policy.max_episodes - self.policy.episode
            self.policy.fit(episodes, reward_loader=lambda: load_net(os.path.join(rewards_dir(), self.environment, self.policy.reward_net_key), True), callbacks=self.policy_callbacks)

    def interrupt(self):
        self.running = False
        self.policy.interrupt()
        if self.train_reward_also:
            self.reward_trainer.interrupt()

    def set_policy(self, policy):
        self.policy = policy


class RewardTrainerThread(Thread):

    def __init__(self, environment, games, reward_train_ended_lock=None):
        super().__init__()
        self.environment = environment
        self.games = games
        self.folder_lock = Semaphore(0)
        self.reward_folder = None
        self.running = True
        self.callbacks = [{"on_train_begin": lambda reward_net: self.set_reward_net_folder(reward_net.folder),
                           "on_epoch_end": lambda reward_net: self.running or reward_net.interrupt()}]
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

    def interrupt(self):
        self.running = False
