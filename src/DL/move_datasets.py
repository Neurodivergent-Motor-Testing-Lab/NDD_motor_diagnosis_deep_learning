import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from random import sample


class SimpleDataset(Dataset):
    def __init__(self, X, y, diagnoses_mappings):
        self.X = X
        self.y = torch.stack((y))

        self.diagnoses_mappings = diagnoses_mappings

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MovementDataset(Dataset):

    def __init__(self, device, X, y, label_shuffle_probability):
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
        if label_shuffle_probability > 0.0:
            unique_labels = np.unique(self.y)
            self.y = self.labelShuffle(unique_labels, label_shuffle_probability)
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

    def labelShuffle(self, unique_labels, probability):
        new_y = self.y.copy()
        for class_label in unique_labels:
            class_indices = np.where(self.y == class_label)[0]
            subset_size = int(class_indices.shape[0] * probability)
            subset_indices = sample(list(class_indices), subset_size)
            new_y[subset_indices] = np.random.choice(unique_labels, subset_size)
        return new_y
