import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import conv_output_size


class RewardNet(nn.Module):
    """ this is a simple neural network """
    
    @staticmethod
    def loss(score_i, score_j): # T-REX loss for a pair of trajectories (or parts of them)
        return nn.CrossEntropyLoss()(torch.stack((score_i, score_j)).reshape(1, 2), torch.tensor([1]))

    def __init__(self, input_shape):
        super(RewardNet, self).__init__()
        self.input_shape = input_shape

        self.conv = nn.Conv2d(in_channels=1, out_channels=15, kernel_size=2)
        o = conv_output_size(input_shape[1], 2, 0, 1)
        self.fc = nn.Linear(15 * o * o, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-5)
        self.batch_size = 5
        # TODO regolarizzare

    def forward(self, x):
        x = x.view(-1, *self.input_shape)
        batch_size = len(x)
        return self.fc(F.relu(self.conv(x)).view(batch_size, -1))

    def fit(self, X_train, max_epochs=1000):
        # training
        for epoch in range(max_epochs):

            # TODO make random subtrajectories

            # give a score to each trajectory
            scores = torch.zeros(len(X_train))
            for t, trajectory in enumerate(X_train):
                trajectory_score = self(trajectory).sum()
                scores[t] = trajectory_score

            # prepare pairs of trajectories scores for loss calculation
            pairs = []
            s = len(scores)
            for i in range(s - 1):
                for j in range(i + 1, s):
                    pairs.append([scores[i], scores[j]])

            # random permute pairs
            permutation = torch.randperm(len(pairs))
            pairs = [pairs[p] for p in permutation]

            # make mini batches
            self.batch_size = self.batch_size if self.batch_size < len(pairs) else len(pairs)
            num_mini_batches = len(pairs) // self.batch_size
            avg_loss = 0
            for b in range(num_mini_batches):

                self.optimizer.zero_grad()
                # for each mini batch, calculate loss and update
                partial_losses = []
                for p in range(self.batch_size):
                    # calculate loss for this pair
                    scores_i, scores_j = pairs[b*self.batch_size + p]
                    partial_losses.append(RewardNet.loss(scores_i, scores_j))

                # calculate total loss of this mini batch
                l = sum(partial_losses)
                # backpropagation
                l.backward(retain_graph=(b < num_mini_batches-1))
                # retain_graph=True is required to not delete stored values during forward pass, because they are needed for next mini batch

                # update net weights
                self.optimizer.step()
                avg_loss += l.item()

            avg_loss /= (num_mini_batches * self.batch_size)
            print("epoch:", epoch, " avg_loss:", avg_loss)

    def evaluate(self, X):
        # net evaluation
        training = self.training
        self.eval()  # same as: self.training = False
        with torch.no_grad():
            test_scores = []
            for t, trajectory in enumerate(X):
                trajectory_score = self(trajectory).sum()
                test_scores.append(trajectory_score)

            quality = 0

            for i in range(len(test_scores)):
                for j in range(len(test_scores)):
                    if i == j:
                        continue
                    print("i: " + str(i) + ", j: " + str(j) + ", P(J(τj) > J(τi)): " + str(test_scores[j] / (test_scores[i] + test_scores[j])))

                    if j > i and test_scores[j] > test_scores[i]:
                            quality += 1

            n = len(test_scores)
            quality /= n * (n-1) / 2
            print("quality:", quality)
            # quality is the percentage of correctly discriminated pairs

        self.training = training  # restore previous training state
