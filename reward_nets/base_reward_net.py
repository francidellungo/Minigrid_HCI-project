import copy
import io
import json
import os
import pickle
from abc import abstractmethod
from contextlib import redirect_stdout
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
from modelsummary import summary

from utils import get_input_shape


class RewardNet(nn.Module):

    @staticmethod
    def loss(score_i, score_j):
        """
        T-REX loss for a pair of trajectories (or parts of them)
        :param (torch.Tensor) score_i: sum of rewards (given by the reward net) for trajectory i
        :param (torch.Tensor) score_j : sum of rewards (given by the reward net) for trajectory j
        :return: (torch.Tensor) T-REX loss (i.e., CrossEntropyLoss with score_j as target)
        """
        pred = torch.stack((score_i, score_j)).reshape(1, 2)
        target = torch.tensor([1]).to(pred.device)
        return nn.CrossEntropyLoss()(pred, target)

    @staticmethod
    def extract_random_subtrajectories_scores(trajectory_i_scores, trajectory_j_scores, subtrajectory_length):
        # define some local variable just to use shorter variable names
        tis = trajectory_i_scores
        tjs = trajectory_j_scores
        sub_len = subtrajectory_length
        li = len(trajectory_i_scores)
        lj = len(trajectory_j_scores)

        if type(sub_len) == tuple:
            assert len(sub_len) == 2
            inf = sub_len[0]
            sup = sub_len[1]
            sub_len = torch.randint(low=inf, high=sup, size=(1,))

        # Control if some of the sub trajectory has length < of the defined sub_length (ai cant be 0-> len = min -1)
        if min(li, lj) < sub_len:
            sub_len = min(li, lj) - 1

        # random choose subtrajectories
        # in the T-REX paper, is ensured that aj>=ai, but in minigrid this seems to not have much sense since better trajectories are usually shorter
        ai = torch.randint(low=0, high=li-sub_len, size=(1,))  # random begin for subtrajectory of trajectory i
        aj = torch.randint(low=0, high=lj-sub_len, size=(1,))  # random begin for subtrajectory of trajectory j


        # for both subtrajectories, return both sums of scores (for T-REX loss) and sums of absolute values of scores (for reward regularization)
        return sum(tis[ai: ai+sub_len]), sum(tjs[aj: aj+sub_len]), sum(tis[ai: ai + sub_len].abs()), sum(tjs[aj: aj + sub_len].abs())

    @staticmethod
    def pad_trajectory(trajectory_scores, num_missing):
        """
        Pads a trajectory appending to it its last state score, num_missing times
        :param (torch.Tensor) trajectory_scores: scores of original (non-padded) trajectory
        :param (int) num_missing: number of times to append the last state score
        :return: (torch.Tensor) scores of padded trajectory
        """
        # if trajectory doesn't need to be padded, return the trajectory
        if num_missing <= 0:
            return trajectory_scores

        last_score = trajectory_scores[-1].reshape(1, 1)
        for i in range(num_missing):
            trajectory_scores = torch.cat((trajectory_scores, last_score))
        return trajectory_scores


    @abstractmethod
    def __init__(self, input_shape, lr=1e-4, folder=None, epoch_to_load=None):  # TODO clean unused arguments
        super(RewardNet, self).__init__()
        self.train_games = None
        if folder is None:
            folder = os.path.curdir
        self.folder = folder
        self.key = os.path.basename(os.path.normpath(folder))
        self.epoch = 0
        self.max_epochs = 0
        self.interrupted = False
        self.running = False

        if epoch_to_load is not None:
            self.load_checkpoint(epoch_to_load)

    @abstractmethod
    def forward(self, x, step=0):
        pass

    def current_device(self):
        return next(self.parameters()).device

    def fit(self, X_train, max_epochs=1000, batch_size=16, num_subtrajectories=7, subtrajectory_length=3, X_val=None, use_also_complete_trajectories=True, train_games_info=None, val_games_info=None, autosave=False, epochs_for_checkpoint=None, train_games=None, callbacks=[]):

        self.train_games = train_games

        ''' print info to open output directory and to open tensorboard '''
        print('output directory:\n"' + os.path.abspath(self.folder) + '"')
        tb_path = os.path.abspath(os.path.join(self.folder, "tensorboard"))
        print('to visualize training progress:\n`tensorboard --logdir="{}"`'.format(tb_path))
        print('creating SummaryWriter...')
        tensorboard = SummaryWriter(tb_path)
        print('SummaryWriter created')

        ''' save info about this training in training.json, and also save the structure of the network '''
        self.save_training_details(batch_size, num_subtrajectories, subtrajectory_length, use_also_complete_trajectories, train_games)
        #torch.save(self, os.path.join(self.folder, "net.pth"))
        pickle.dump(self, open(os.path.join(self.folder, "net.pkl"), "wb"))

        # TODO ha senso che subtrajectory_length invece di una costante sia un range entro il quale scegliere a random la lunghezza della sottotraiettoria?
        # TODO bisogna capire quale è un buon modo per scegliere tutti questi iperparametri delle sottotraiettorie
        
        for callback in callbacks:
            if "on_train_begin" in callback:
                callback["on_train_begin"](self)

        ''' begin training '''
        self.running = True
        for epoch in range(max_epochs):

            ''' forward pass on the whole training set '''
            # give a score to each train trajectory
            train_trajectories_scores = self.calculate_trajectories_scores(X_train)

            ''' prepare pairs of trajectories scores for loss calculation '''
            pairs = []          # needed for T-REX loss
            abs_rewards = []    # needed for reward regularization
            s = len(train_trajectories_scores)
            for i in range(s - 1):
                for j in range(i + 1, s):
                    # TODO è corretto considerare sempre tutte le possibili coppie? Oppure anche le coppie dovrebbero essere scelte a random in ogni epoca?
                    ''' for each pair of trajectories, select num_subtrajectories random subtrajectories '''
                    for k in range(num_subtrajectories):
                        sum_i, sum_j, sum_abs_i, sum_abs_j = RewardNet.extract_random_subtrajectories_scores(train_trajectories_scores[i], train_trajectories_scores[j], subtrajectory_length)
                        pairs.append([sum_i, sum_j])
                        abs_rewards.append([sum_abs_i, sum_abs_j])

                    # TODO consideriamo anche le traiettorie complete oppure solo le sottotraiettorie random?
                    ''' consider full trajectories '''
                    # queste due righe sotto sono per le traiettorie complete senza padding
                    pairs.append([sum(train_trajectories_scores[i]), sum(train_trajectories_scores[j])])
                    abs_rewards.append([sum(train_trajectories_scores[i].abs()), sum(train_trajectories_scores[j].abs())])

                    # queste due righe sotto sono per le traiettorie complete con padding
                    li = len(train_trajectories_scores[i])
                    lj = len(train_trajectories_scores[j])
                    if li > lj:
                        pi = train_trajectories_scores[i]
                        pj = RewardNet.pad_trajectory(train_trajectories_scores[j], li - lj)
                        pairs.append([sum(pi), sum(pj)])
                        abs_rewards.append([sum(pi.abs()), sum(pj.abs())])

            ''' shuffle pairs and make mini batches'''
            # random permute pairs
            permutation = torch.randperm(len(pairs)).tolist()
            pairs = [pairs[p] for p in permutation]
            abs_rewards = [abs_rewards[p] for p in permutation]

            # make mini batches
            batch_size = batch_size if batch_size < len(pairs) else len(pairs)
            num_mini_batches = len(pairs) // batch_size
            avg_batch_loss = 0

            for b in range(num_mini_batches):

                self.optimizer.zero_grad()
                # for each mini batch, calculate loss and update
                partial_losses = []
                partial_abs_rewards = []
                for p in range(batch_size):
                    ''' calculate loss for this pair '''
                    scores_i, scores_j = pairs[b*batch_size + p]
                    partial_losses.append(RewardNet.loss(scores_i, scores_j))
                    partial_abs_rewards.extend(abs_rewards[b*batch_size + p])

                # calculate total loss of this mini batch
                l = sum(partial_losses)/batch_size + self.lambda_abs_rewards * sum(partial_abs_rewards)/batch_size  # L1 rewards regularization

                ''' backpropagation and optimizer step to update net weights '''
                l.backward(retain_graph=(b < num_mini_batches-1))
                # retain_graph=True is required to not delete (after backpropagation) values stored during the forward pass, because they are needed for the backpropagation of the next mini batch
                self.optimizer.step()

                avg_batch_loss += l.item()

            ''' update, print and save metrics '''
            # calculate training metrics
            avg_batch_loss /= num_mini_batches
            train_corr = self.correlation(train_trajectories_scores, train_games_info)
            train_quality = self.quality(X_train)

            epoch_summary = "epoch: {},  avg_batch_loss: {:7.4f},  correlation_on_train: {:7.4f},  quality_on_train: {:7.4f}".\
                format(epoch, avg_batch_loss, train_corr, train_quality)

            # save training metrics
            tensorboard.add_scalar('average training batch loss', avg_batch_loss, epoch)
            tensorboard.add_scalars('correlation', {"train": train_corr}, epoch)
            tensorboard.add_scalars('quality', {"train": train_quality}, epoch)

            # check if a validation set is specified
            if X_val is not None:
                if val_games_info is not None:
                    # calculate and save correlation on validation set, if groundtruth reward is specified
                    val_trajectories_scores = self.calculate_trajectories_scores(X_val)
                    val_corr = self.correlation(val_trajectories_scores, val_games_info)
                    epoch_summary += ",  correlation_on_val: {:7.4f}".format(val_corr)
                    tensorboard.add_scalars('correlation', {"val": val_corr}, epoch)

                # calculate and save quality on validation set
                val_quality = self.quality(X_val)
                epoch_summary += ",  quality_on_val: {:7.4f}".format(val_quality)
                tensorboard.add_scalars('quality', {"val": val_quality}, epoch)

            # print metrics
            print(epoch_summary)

            # check if I have to save the net weights in this epoch
            if autosave:
                if (epochs_for_checkpoint is not None and epoch % epochs_for_checkpoint == epochs_for_checkpoint - 1) or epoch == max_epochs-1:
                    # save net weights
                    self.save_checkpoint(epoch)
                
            for callback in callbacks:
                if "on_epoch_end" in callback:
                    callback["on_epoch_end"](self)

            if self.interrupted:
                break

        self.running = False

        ''' training ended '''
        tensorboard.close()

        for callback in callbacks:
            if "on_train_end" in callback:
                callback["on_train_end"](self)

    ''' calculate scores for trajectories '''
    def calculate_trajectories_scores(self, X):
        trajectories_scores = []
        for t, trajectory in enumerate(X):
            trajectory_scores = self(trajectory, torch.tensor(range(len(trajectory)), requires_grad=False))  # TODO calcolare per più traiettorie alla volta?
            trajectories_scores.append(trajectory_scores)

        return trajectories_scores

    ''' calculate quality metric'''
    def quality(self, X):
        test_scores = []
        for t, trajectory in enumerate(X):
            trajectory_score = self(trajectory, torch.tensor(range(len(trajectory)), requires_grad=False)).sum()
            test_scores.append(trajectory_score)

        quality = 0

        for i in range(len(test_scores)):
            for j in range(len(test_scores)):
                if i == j:
                    continue
                # print("i: " + str(i) + ", j: " + str(j) + ", P(J(τj) > J(τi)): " + str(
                #     test_scores[j].exp() / (test_scores[i].exp() + test_scores[j].exp() + 10 ** -7)))

                if j > i and test_scores[j] > test_scores[i]:
                    quality += 1

        n = len(test_scores)
        quality /= n * (n - 1) / 2
        # quality is the fraction of correctly discriminated pairs
        return quality

    ''' calculate correlation metric '''
    def correlation(self, trajectories_scores, games_info):

        true_trajectories_scores = [x["score"] for x in games_info]

        true_mx = max(true_trajectories_scores)
        true_mn = min(true_trajectories_scores)
        assert true_mx >= true_mn

        trajectories_scores = [sum(scores).item() for scores in trajectories_scores]
        mx = max(trajectories_scores)
        mn = min(trajectories_scores)
        assert mx >= mn

        if mx == mn:
            return np.nan

        normalized_trajectories_scores = [(s-mn)/(mx-mn)*(true_mx-true_mn) + true_mn for s in trajectories_scores]
        # print("corr")
        # print(np.corrcoef(true_trajectories_scores, trajectories_scores))
        # print(np.corrcoef(true_trajectories_scores, normalized_trajectories_scores))
        # plt.scatter(true_trajectories_scores, trajectories_scores)
        # plt.show()
        return np.corrcoef(true_trajectories_scores, normalized_trajectories_scores)[0][1]

    ''' save net and training details in training.json '''
    def save_training_details(self, batch_size, num_subtrajectories, subtrajectory_length, use_also_complete_trajectories, train_games):
        print("saving details")
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
            'created dir "' + self.folder + '"'

        with open(os.path.join(self.folder, "training.json"), "wt") as file:
            with io.StringIO() as out, redirect_stdout(out):
                summary(self, torch.zeros(get_input_shape()).to(self.current_device()), show_input=True)
                summary(self, torch.zeros(get_input_shape()).to(self.current_device()), show_input=False)
                net_summary = out.getvalue()
            print(net_summary)
            j = {"type": str(type(self)), "str": str(self).replace("\n", ""), "optimizer": str(self.optimizer),
                 "penalty_rewards": self.lambda_abs_rewards, "batch_size": batch_size,
                 "num_subtrajectories": num_subtrajectories, "subtrajectory_length": subtrajectory_length,
                 "use_also_complete_trajectories": use_also_complete_trajectories, "summary": net_summary,
                 "max_epochs": self.max_epochs}
            if train_games is not None:
                j["games"] = train_games
            json.dump(j, file, indent=True)
            print('details saved on ' + os.path.join(self.folder, "training.json"))

    def save_network(self):
        # torch.save(self, os.path.join(self.folder, "net.pth"))
        with open(os.path.join(self.folder, "net.pkl"), "wb") as file:
            pickle.dump(self, file)
        return self

    ''' save net weights (remark: only weights are saved here, not the network structure!) '''
    def save_checkpoint(self, epoch):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        checkpoint_file = os.path.join(self.folder, "reward_net-" + str(epoch) + ".pth")
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, epoch_to_load_weights):
        # epoch_to_load_weights = number of the epoch
        if epoch_to_load_weights is not None:
            self.load_state_dict(
                torch.load(os.path.join(self.folder, "reward_net-" + str(epoch_to_load_weights) + ".pth"),
                           map_location=self.current_device()))

            self.epoch = epoch_to_load_weights
            self.max_epochs = self._get_last_training_max_epochs()
        else:
            print("Error: cannot find saved weights")
        return self

    def load_last_checkpoint(self):
        # load the most recent weights from the folder of this policy
        epoch_to_load_weights = self._get_last_saved_reward_epoch()
        return self.load_checkpoint(epoch_to_load_weights)

    def _get_last_saved_reward_epoch(self):
        epochs_saved_weights = [int(state.rsplit("-", 1)[1].split(".", 1)[0]) for state in glob(os.path.join(self.folder, "reward_net-*.pth"))]
        if len(epochs_saved_weights) > 0:
            return max(epochs_saved_weights)
        return None

    def _get_last_training_max_epochs(self):
        with open(os.path.join(self.folder, "training.json"), "rt") as file:
            j = json.load(file)
        return j["max_epochs"]

    def interrupt(self):
        self.interrupted = True

