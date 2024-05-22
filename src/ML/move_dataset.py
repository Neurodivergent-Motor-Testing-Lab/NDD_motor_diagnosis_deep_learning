import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing


class MovementDataset(Dataset):

    def __init__(self, device, X, y):
        self.device = device
        self.X = X
        le = preprocessing.LabelEncoder()
        le.fit(y)
        self.groups = le.transform(y)

        diagnoses = []
        for user_str in y:
            diagnoses.append(user_str.split("-")[0])
        diagnoses = np.asarray(diagnoses)

        le = preprocessing.LabelEncoder()
        le.fit(diagnoses)
        self.y = le.transform(diagnoses)
        self.diagnoses_mappings = dict(zip(le.classes_, le.transform(le.classes_)))
        unique, label_counts = np.unique(self.y, return_counts=True)
        dataset_counts = dict(zip(unique, label_counts))
        print(self.diagnoses_mappings)
        for diagnosis in self.diagnoses_mappings:
            print(
                "Count of ",
                diagnosis,
                "(",
                self.diagnoses_mappings[diagnosis],
                ") is ",
                dataset_counts[self.diagnoses_mappings[diagnosis]],
            )

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        xs = self.X[idx, :, :]
        xs = torch.from_numpy(xs).float().to(device=self.device)
        xs = xs.transpose(0, 1)
        ys = np.atleast_1d(self.y[idx])
        ys = torch.from_numpy(ys).float().type(torch.LongTensor).to(device=self.device)

        return xs, ys
