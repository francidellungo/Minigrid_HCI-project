import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class RewardNet(nn.Module):
    """ this is a simple neural network """

    @staticmethod
    def calculateOutputSize(input_dim, filter_dim, padding=0, stride=1):
        # formula for output dimension:
        # O = (D -K +2P)/S + 1
        # where:
        #   D = input dim (height/length)
        #   K = filter size
        #   P = padding
        #   S = stride
        return (input_dim - filter_dim + 2*padding) // stride + 1
    
    @staticmethod
    def loss(scores):
        n = len(scores)
        scores = scores + 10 ** -3

        predictions = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                predictions.append((scores[j]) / (scores[i] + scores[j]))  # T-REX loss
        l = -sum([p.log() for p in predictions])
        return l
    
    def __init__(self, input_shape):
        super(RewardNet, self).__init__()
        assert input_shape[1] == input_shape[2]

        self.conv = nn.Conv2d(in_channels=1, out_channels=15, kernel_size=2)
        o = RewardNet.calculateOutputSize(input_shape[1], 2, 0, 1)
        self.fc = nn.Linear(15 * o * o, 1)
        #self.optimizer = optim.Adadelta(self.parameters(), lr=1e-4)
        #self.optimizer = optim.Adamax(self.parameters(), lr=1e-4)
        #self.optimizer = optim.RMSprop(self.parameters(), lr=1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        #self.optimizer = optim.ASGD(self.parameters(), lr=1e-4)
        #self.optimizer = optim.Rprop(self.parameters(), lr=1e-4)
        #self.optimizer = optim.SGD(self.parameters(), lr=1e-4)

    def forward(self, x):
        b = x.shape[0]
        return self.fc(F.relu(self.conv(x)).view(b, -1))

    def fit(self, X_train, max_epochs=1000):
        # TRAINING
        for epoch in range(max_epochs):
            self.optimizer.zero_grad()
            scores = torch.zeros(len(X_train))
            for t, trajectory in enumerate(X_train):
                trajectory_score = self(trajectory).sum().clamp(max=50).exp()  # TODO il clamp servirebbe per evitare un input troppo grande all'exp, ma a volte qualcosa va storto comunque
                scores[t] = trajectory_score

            l = RewardNet.loss(scores)
            l.backward()

            self.optimizer.step()

            print("epoch:", epoch, " loss:", l.item())

    def evaluate(self, X):
        # TEST BEFORE TRAINING
        with torch.no_grad():
            test_scores = []
            for t, trajectory in enumerate(X):
                trajectory_score = self(trajectory).sum().exp()
                test_scores.append(trajectory_score)
    
            for i in range(len(test_scores)):
                for j in range(len(test_scores)):
                    if i == j:
                        continue
                    print("i: " + str(i) + ", j: " + str(j) + ", P(J(τj) > J(τi)): " + str(test_scores[j] / (test_scores[i] + test_scores[j])))
