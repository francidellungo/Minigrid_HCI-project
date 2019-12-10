import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import conv_output_size


class RewardNet(nn.Module):
    
    @staticmethod
    def loss(score_i, score_j): # T-REX loss for a pair of trajectories (or parts of them)
        pred = torch.stack((score_i, score_j)).reshape(1, 2)
        target = torch.tensor([1]).to(pred.device)
        return nn.CrossEntropyLoss()(pred, target)

    @staticmethod
    def extract_random_subtrajectories_scores(trajectory_i_scores, trajectory_j_scores, subtrajectory_length):
        # define some local variable just to use shorter variable names
        tis = trajectory_i_scores
        tjs = trajectory_j_scores
        l = subtrajectory_length
        li = len(trajectory_i_scores)
        lj = len(trajectory_j_scores)

        # random choose subtrajectories
        # in the T-REX paper, is ensured that aj>=ai, but in minigrid this seems to not have much sense since better trajectories are usually shorter
        ai = torch.randint(low=0, high=li-subtrajectory_length, size=(1,))  # random begin for subtrajectory of trajectory i
        aj = torch.randint(low=0, high=lj-subtrajectory_length, size=(1,))  # random begin for subtrajectory of trajectory j

        # for both subtrajectories, return both sums of scores (for T-REX loss) and sums of absolute values of scores (for reward regularization)
        return sum(tis[ai: ai+l]), sum(tjs[aj: aj+l]), sum(tis[ai: ai + l].abs()), sum(tjs[aj: aj + l].abs())

    def __init__(self, input_shape, lr=1e-4):
        super(RewardNet, self).__init__()
        self.input_shape = input_shape

        # simple net with: 2D convolutional layer -> activation layer -> fully connected layer
        self.conv = nn.Conv2d(in_channels=1, out_channels=15, kernel_size=2)
        o = conv_output_size(input_shape[1], 2, 0, 1)
        self.fc = nn.Linear(15 * o * o, 1)

        # regularization
        weight_decay = 10 ** -4  # penalty for net weights L2 regularization
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.lambda_abs_rewards = 10 ** -3  # penalty for rewards regularization

        # TODO tutti questi iperparametri dovrebbero essere presi come parametri in ingresso
        # TODO tutti questi iperparametri sono completamente ad occhio: vanno scelti per bene (ma in che modo?)
        # TODO la loro scelta dipende da varie cose, ad esempio se consideriamo o meno anche le traiettorie complete nel calolo della loss oppure solo quelle parziali


    def forward(self, x):
        x = x.view(-1, *self.input_shape)
        batch_size = len(x)
        return self.fc(F.relu(self.conv(x)).view(batch_size, -1))

    def fit(self, X_train, max_epochs=1000, batch_size=16, num_subtrajectories=5, subtrajectory_length=4):

        # TODO ha senso che subtrajectory_length invece di una costante sia un range entro il quale scegliere a random la lunghezza della sottotraiettoria?
        # TODO bisogna capire quale è un buon modo per scegliere tutti questi iperparametri delle sottotraiettorie

        # t_lens = []
        # for trajectory in X_train:
        #     t_lens.append(len(trajectory))
        #
        # X_train = torch.cat([trajectory for trajectory in X_train])

        # training
        for epoch in range(max_epochs):

            # give a score to each trajectory
            trajectories_scores = []
            for t, trajectory in enumerate(X_train):
                trajectory_scores = self(trajectory)
                trajectories_scores.append(trajectory_scores)

            # TODO queste righe sotto commentate (e le righe prima del for) sono un tentativo di sfruttare meglio la potenza di calcolo della GPU, ma per ora non sembrano avere una particolare influenza
            # scores = self(X_train)
            # trajectories_scores = []
            # begin = 0
            # for t_len in t_lens:
            #     trajectories_scores.append(scores[begin:begin+t_len])
            #     begin += t_len


            # prepare pairs of trajectories scores for loss calculation
            pairs = []          # needed for T-REX loss
            abs_rewards = []    # needed for reward regularization
            s = len(trajectories_scores)
            for i in range(s - 1):
                for j in range(i + 1, s):
                    # TODO è corretto considerare sempre tutte le possibili coppie? Oppure anche le coppie dovrebbero essere scelte a random in ogni epoca?
                    # for each pair of trajectories, select num_subtrajectories random subtrajectories
                    for k in range(num_subtrajectories):
                        sum_i, sum_j, sum_abs_i, sum_abs_j = RewardNet.extract_random_subtrajectories_scores(trajectories_scores[i], trajectories_scores[j], subtrajectory_length)
                        pairs.append([sum_i, sum_j])
                        abs_rewards.append([sum_abs_i, sum_abs_j])

                    # TODO consideriamo anche le traiettorie complete oppure solo le sottotraiettorie random?
                    pairs.append([sum(trajectories_scores[i]), sum(trajectories_scores[j])])
                    abs_rewards.append([sum(trajectories_scores[i].abs()), sum(trajectories_scores[j].abs())])

            # random permute pairs
            permutation = torch.randperm(len(pairs))
            pairs = [pairs[p] for p in permutation]
            abs_rewards = [abs_rewards[p] for p in permutation]

            # make mini batches
            batch_size = batch_size if batch_size < len(pairs) else len(pairs)
            num_mini_batches = len(pairs) // batch_size
            avg_loss = 0
            for b in range(num_mini_batches):

                self.optimizer.zero_grad()
                # for each mini batch, calculate loss and update
                partial_losses = []
                partial_abs_rewards = []
                for p in range(batch_size):
                    # calculate loss for this pair
                    scores_i, scores_j = pairs[b*batch_size + p]
                    partial_losses.append(RewardNet.loss(scores_i, scores_j))
                    partial_abs_rewards.extend(abs_rewards[b*batch_size + p])

                # calculate total loss of this mini batch
                l = sum(partial_losses)/batch_size + self.lambda_abs_rewards * (sum(partial_abs_rewards) + sum([r*r for r in partial_abs_rewards]))/batch_size  # rewards regularization is L1 + L2 regularization

                # backpropagation
                l.backward(retain_graph=(b < num_mini_batches-1))
                # retain_graph=True is required to not delete stored values during forward pass, because they are needed for next mini batch

                # update net weights
                self.optimizer.step()
                avg_loss += l.item()

            avg_loss /= num_mini_batches
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
                    print("i: " + str(i) + ", j: " + str(j) + ", P(J(τj) > J(τi)): " + str(test_scores[j].exp() / (test_scores[i].exp() + test_scores[j].exp() + 10 ** -7)))

                    if j > i and test_scores[j] > test_scores[i]:
                            quality += 1

            n = len(test_scores)
            quality /= n * (n-1) / 2
            print("quality:", quality)
            # quality is the percentage of correctly discriminated pairs

        self.training = training  # restore previous training state
