import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from livelossplot import PlotLosses
import tqdm

from spectroscope.data import SubsetsLoader, get_filtered_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Metrics:
    def __init__(self, per_label_train_loader, per_label_test_loader, loss_fn):
        self.per_label_train_loader = per_label_train_loader
        self.per_label_test_loader = per_label_test_loader
        self.loss_fn = loss_fn

        self.train_loss = np.zeros(10)
        self.train_accuracy = np.zeros(10)
        self.test_loss = np.zeros(10)
        self.test_accuracy = np.zeros(10)

        self.trainset_sizes = np.array([len(subset) for subset in per_label_train_loader.subsets])
        self.testset_sizes = np.array([len(subset) for subset in per_label_test_loader.subsets])

    def measure(self, model: nn.Module):
        with torch.no_grad():
            # Loop over each specific-label-restricted subset of data
            for l in tqdm(range(10), desc="Measuring metrics"):
                self.per_label_train_loader.to_subset(l)
                self.per_label_test_loader.to_subset(l)

                for i, (x, y) in enumerate(self.per_label_train_loader):
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    y_pred = model(x)
                    _, predicted = torch.max(y_pred.data, 1)
                    self.train_accuracy[l] += (predicted == y).sum().item()
                    self.train_loss[l] += self.loss_fn(y_pred, y).item()

                self.train_accuracy[l] /= self.trainset_sizes[l]
                self.train_loss[l] /= self.trainset_sizes[l]

                for i, (x, y) in enumerate(self.per_label_test_loader):
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    y_pred = model(x)
                    _, predicted = torch.max(y_pred.data, 1)
                    self.test_accuracy[l] += (predicted == y).sum().item()
                    self.test_loss[l] += self.loss_fn(y_pred, y).item()

                self.test_accuracy[l] /= self.testset_sizes[l]
                self.test_loss[l] /= self.testset_sizes[l]

    def reset(self):
        self.train_loss = np.zeros(10)
        self.train_accuracy = np.zeros(10)
        self.test_loss = np.zeros(10)
        self.test_accuracy = np.zeros(10)

    @property
    def total_train_size(self):
        return self.trainset_sizes.sum() 
    
    @property
    def total_test_size(self):
        return self.testset_sizes.sum()

    @property
    def total_train_loss(self):
        return (self.train_loss * self.trainset_sizes).sum() / self.total_train_size
    
    @property
    def total_test_loss(self):
        return (self.test_loss * self.testset_sizes).sum() / self.total_test_size

    @property
    def total_train_accuracy(self):
        return (self.train_accuracy * self.trainset_sizes).sum() / self.total_train_size
    
    @property
    def total_test_accuracy(self):
        return (self.test_accuracy * self.testset_sizes).sum() / self.total_test_size

    def as_dict(self):
        d = {
            "train/loss/total": self.total_train_loss,
            "test/loss/total": self.total_test_loss,
            "train/accuracy/total": self.total_train_accuracy,
            "test/accuracy/total": self.total_test_accuracy,
        }

        for l in range(10):
            d[f"train/loss/{l}"] = self.train_loss[l]
            d[f"test/loss/{l}"] = self.test_loss[l]
            d[f"train/accuracy/{l}"] = self.train_accuracy[l]
            d[f"test/accuracy/{l}"] = self.test_accuracy[l]

        return d      