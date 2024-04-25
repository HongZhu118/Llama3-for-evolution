import numpy as np
import random


"""
 Restricted Boltzmann Machine
"""


class RBM:

    def __init__(self, nVisible=0, nHidden=0, Epoch=10, BatchSize=1, Penalty=0.01, Momentum=0.5,  # noqa
                 LearnRate=0.1, Weight=None, vBias=None, hBias=None, ) -> None:
        self.nVisible = nVisible
        self.nHidden = nHidden
        self.Epoch = Epoch
        self.BatchSize = BatchSize
        self.Penalty = Penalty
        self.Momentum = Momentum
        self.LearnRate = LearnRate
        if Weight is None:
            self.Weight = 0.1 * np.random.randn(self.nVisible, self.nHidden)
        else:
            self.Weight = Weight
        if vBias is None:
            self.vBias = np.zeros(self.nVisible)
        else:
            self.vBias = vBias
        if hBias is None:
            self.hBias = np.zeros(self.nHidden)
        else:
            self.hBias = hBias
    """
    Train
    """
    def train(self, X):
        vishidinc = np.zeros(self.Weight.shape)
        hidbiasinc = np.zeros(self.hBias.shape)
        visbiasinc = np.zeros(self.vBias.shape)
        for i in range(self.Epoch):
            if self.Epoch > 5:
                self.Momentum = 0.9
            kk = random.sample(np.arange(X.shape[0]).tolist(), X.shape[0])
            for batch in range(int(X.shape[0] / self.BatchSize)):
                batchdata = X[kk[batch *
                                 self.BatchSize: (batch + 1) * self.BatchSize], :]  # noqa
                """
                Positive phase
                """
                poshidprobs = 1 / (1 + np.exp(np.dot(-batchdata.astype(int),
                                   self.Weight) - np.tile(self.hBias, (self.BatchSize, 1))))  # noqa
                poshidstates = poshidprobs > np.random.random(
                    size=(self.BatchSize, self.nHidden))
                """
                Negative phase
                """
                negdataprobs = 1 / (1 + np.exp(np.dot(-poshidstates.astype(int),  # noqa
                                    self.Weight.T) - np.tile(self.vBias, (self.BatchSize, 1))))  # noqa
                negdata = negdataprobs > np.random.random(
                    size=(self.BatchSize, self.nVisible))
                neghidprobs = 1 / (1 + np.exp(np.dot(-negdata.astype(int),
                                   self.Weight) - np.tile(self.hBias, (self.BatchSize, 1))))  # noqa
                """
                Update weight
                """
                posprods = np.dot(batchdata.T, poshidprobs)
                negprods = np.dot(negdataprobs.T, neghidprobs)
                poshidact = np.sum(poshidprobs)
                posvisact = np.sum(batchdata)
                neghidact = np.sum(neghidprobs)
                negvisact = np.sum(negdata)
                vishidinc = self.Momentum * vishidinc + self.LearnRate * \
                    ((posprods - negprods) / self.BatchSize -
                     self.Penalty * self.Weight)
                visbiasinc = self.Momentum * visbiasinc + \
                    (self.LearnRate / self.BatchSize) * (posvisact - negvisact)
                hidbiasinc = self.Momentum * hidbiasinc + \
                    (self.LearnRate / self.BatchSize) * (poshidact - neghidact)
                self.Weight = self.Weight + vishidinc
                self.vBias = self.vBias + visbiasinc
                self.hBias = self.hBias + hidbiasinc

    def reduce(self, X):
        return 1 / (1 + np.exp(np.dot(-X.astype(int), self.Weight) - np.tile(self.hBias, (X.shape[0], 1)))) > \
            np.random.random(size=(X.shape[0], self.Weight.shape[1]))  # noqa

    def recover(self, X):
        return 1 / (1 + np.exp(np.dot(-X.astype(int), self.Weight.T) - np.tile(self.vBias, (X.shape[0], 1)))) > \
            np.random.random(size=(X.shape[0], self.Weight.shape[0]))  # noqa
