import json
import os
import h5py
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
import soundfile as sf
from config import *
EPSILON = 1e-10

class To_Tensor(object):
    def __call__(self, x, type='float'):
        if type == 'float':
            return torch.FloatTensor(x)
        elif type == 'int':
            return  torch.IntTensor(x)

class TrainDataset(Dataset):
    def __init__(self, json_dir, batch_size):
        self.json_dir = json_dir
        self.batch_size = batch_size
        json_pos= os.path.join(json_dir, 'train', 'files.json')
        with open(json_pos, 'r') as f:
            json_list = json.load(f)

        minibatch = []
        start = 0
        while True:
            end = min(len(json_list), start+ batch_size)
            minibatch.append(json_list[start:end])
            start = end
            if end == len(json_list):
                break
        self.minibatch = minibatch

    def __len__(self):
        return len(self.minibatch)

    def __getitem__(self, index):
        return self.minibatch[index]

class CvDataset(Dataset):
    def __init__(self, json_dir, batch_size):
        self.json_dir = json_dir
        self.batch_size = batch_size
        json_pos= os.path.join(json_dir, 'cv', 'files.json')
        with open(json_pos, 'r') as f:
            json_list = json.load(f)

        minibatch = []
        start = 0
        while True:
            end = min(len(json_list), start+ batch_size)
            minibatch.append(json_list[start:end])
            start = end
            if end == len(json_list):
                break
        self.minibatch = minibatch

    def __len__(self):
        return len(self.minibatch)

    def __getitem__(self, index):
        return self.minibatch[index]

class TrainDataLoader(object):
    def __init__(self, data_set, **kw):
        self.data_loader = DataLoader(dataset=data_set,
                                      shuffle=1,
                                      collate_fn=self.collate_fn,
                                      **kw)

    @staticmethod
    def collate_fn(batch):
        feats, labels, frame_mask_list = generate_feats_labels(batch)
        return BatchInfo(feats, labels, frame_mask_list)

    def get_data_loader(self):
        return self.data_loader

def generate_feats_labels(batch):

    batch = batch[0]

    feat_list = []
    label_list = []

    for idx in range(len(batch)):

        name = batch[idx]

        noisy = np.load(
            os.path.join(
                file_path,
                "noisy",
                name + ".npy"
            )
        )

        clean = np.load(
            os.path.join(
                file_path,
                "clean",
                name + ".npy"
            )
        )

        noisy = torch.tensor(
            noisy,
            dtype=torch.float32
        )

        clean = torch.tensor(
            clean,
            dtype=torch.float32
        )

        feat_list.append(noisy)
        label_list.append(clean)

    feat_list = torch.stack(
        feat_list,
        dim=0
    )

    label_list = torch.stack(
        label_list,
        dim=0
    )

    return feat_list, label_list, None

def cv_generate_feats_labels(batch):

    batch = batch[0]

    feat_list = []
    label_list = []

    for idx in range(len(batch)):

        name = batch[idx]

        noisy = np.load(
            os.path.join(
                file_path,
                "val_noisy",
                name + ".npy"
            )
        )

        clean = np.load(
            os.path.join(
                file_path,
                "val_clean",
                name + ".npy"
            )
        )

        feat_list.append(
            torch.tensor(
                noisy,
                dtype=torch.float32
            )
        )

        label_list.append(
            torch.tensor(
                clean,
                dtype=torch.float32
            )
        )

    feat_list = torch.stack(feat_list)

    label_list = torch.stack(label_list)

    return feat_list, label_list, None


class CvDataLoader(object):
    def __init__(self, data_set, **kw):
        self.data_loader = DataLoader(dataset=data_set,
                                      shuffle=1,
                                      collate_fn=self.collate_fn,
                                      **kw)

    @staticmethod
    def collate_fn(batch):
        feats, labels, frame_mask_list = cv_generate_feats_labels(batch)
        return BatchInfo(feats, labels, frame_mask_list)

    def get_data_loader(self):
        return self.data_loader

class BatchInfo(object):
    def __init__(self, feats, labels, frame_mask_list):
        self.feats = feats
        self.labels = labels
        self.frame_mask_list = frame_mask_list