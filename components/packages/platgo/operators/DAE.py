import numpy as np
import random


"""
 Feedforward neural network
"""


class DAE:

    def __init__(self, nVisible=0, nHidden=0, Epoch=10, BatchSize=1, InputZeroMaskedFraction=0.5, Momentum=0.5,  # noqa
                 LearnRate=0.1, WA=None, WB=None, lower=None, upper=None) -> None:  # noqa
        self.nVisible = nVisible
        self.nHidden = nHidden
        self.Epoch = Epoch
        self.BatchSize = BatchSize
        self.InputZeroMaskedFraction = InputZeroMaskedFraction
        self.Momentum = Momentum
        self.LearnRate = LearnRate
        self.lower = lower
        self.upper = upper
        if WA is None:
            self.WA = (np.random.random(size=(self.nHidden, self.nVisible + 1)
                                        ) - 0.5) * 8 * np.sqrt(6 / (self.nHidden + self.nVisible))  # noqa
        else:
            self.WA = WA
        if WB is None:
            self.WB = (np.random.random(size=(self.nVisible, self.nHidden + 1)
                                        ) - 0.5) * 8 * np.sqrt(6 / (self.nVisible + self.nHidden))  # noqa
        else:
            self.WB = WB

    def train(self, X):
        self.lower = np.min(X, axis=0)
        self.upper = np.max(X, axis=0)
        X = (X - np.tile(self.lower, (X.shape[0], 1))) / \
            np.tile(self.upper - self.lower, (X.shape[0], 1))
        vW = [[], []]
        vW[0] = np.zeros(self.WA.shape)
        vW[1] = np.zeros(self.WB.shape)
        if self.InputZeroMaskedFraction != 0:
            theta = np.random.random(
                size=X.shape) > self.InputZeroMaskedFraction
        else:
            theta = np.ones(X.shape).astype(bool)
        X_temp = X * theta
        X_temp = np.hstack((np.ones((X.shape[0], 1)), X_temp))
        for i in range(self.Epoch):
            kk = random.sample(np.arange(X.shape[0]).tolist(), X.shape[0])
            for batch in range(int(X.shape[0] / self.BatchSize)):
                batch_x = X_temp[kk[batch *
                                    self.BatchSize: (batch + 1) * self.BatchSize], :]  # noqa
                batch_y = X[kk[batch *
                               self.BatchSize: (batch + 1) * self.BatchSize], :]  # noqa

                # Feedforward pass
                poshid1 = 1 / (1 + np.exp(np.dot(-batch_x, self.WA.T)))
                poshid1 = np.hstack((np.ones((self.BatchSize, 1)), poshid1))
                poshid2 = 1 / (1 + np.exp(np.dot(-poshid1, self.WB.T)))

                # BP
                e = batch_y - poshid2
                d = [[], [], []]
                dW = [[], []]
                d[2] = -e * (poshid2 * (1 - poshid2))
                d_act = poshid1 * (1 - poshid1)
                d[1] = np.dot(d[2], self.WB) * d_act
                for j in range(2):
                    if j + 1 == 2:
                        dW[j] = np.dot(d[j + 1].T, poshid1) / d[2].shape[0]
                    else:
                        dW[j] = np.dot(d[j + 1][:, 1:].T,
                                       batch_x) / d[j + 1].shape[0]
                for j in range(2):
                    dW[j] = self.LearnRate * dW[j]
                    if self.Momentum > 0:
                        vW[j] = self.Momentum * vW[j] + dW[j]
                        dW[j] = vW[j]
                    if j == 0:
                        self.WA = self.WA - dW[j]
                    else:
                        self.WB = self.WB - dW[j]

    def reduce(self, X):
        X = (X - np.tile(self.lower, (X.shape[0], 1))) / \
            np.tile(self.upper - self.lower, (X.shape[0], 1))
        return 1 / (1 + np.exp(np.dot(-X, self.WA[:, 1:].T) - np.tile(self.WA[:, 0].T, (X.shape[0], 1))))  # noqa

    def recover(self, H):
        X = 1 / (1 + np.exp(np.dot(-H,
                 self.WB[:, 1:].T) - np.tile(self.WB[:, 0].T, (H.shape[0], 1))))  # noqa
        X = X * np.tile(self.upper - self.lower,
                        (X.shape[0], 1)) + np.tile(self.lower, (X.shape[0], 1))
        return X
