# -*- coding: utf-8 -*-
"""
MT 数据加载器（对应语音 DBT-Net 的数据流）：

连续干净波形
        -> 按源划分 train/cv（prepare_mt_data.py 完成：6 train / 2 cv）
        -> 随机截取变长片段（4096~16384 点，T≈17~65）
           train：每次取样本重新随机起点；cv：固定起点
        -> 时域随机加噪（训练时随机 SNR，验证时固定 SNR）
        -> STFT -> RI 特征 [B, 2, T, F=257]

训练时每个 epoch 的 minibatch 顺序随机打乱（shuffle=True），
且每次取到样本后重新随机截取起点、随机加噪，等价于在线数据增广。
"""
import hashlib
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config_mt import (
    json_dir,
    pool_dir,
    batch_size,
    FS,
    N_FFT,
    HOP,
    WINDOW,
    SNR_RANGE,
    CV_SNR,
)
from mt_stft import mt_stft
from mt_noise import generate_mt_noise, mix_noisy

EPSILON = 1e-12


# 数据池缓存（每个进程只读一次）
_POOL_CACHE = {}


def _load_pool_for(inst, split_name):
    key = (split_name, inst["channel"], inst["source"])
    if key not in _POOL_CACHE:
        _POOL_CACHE[key] = np.load(
            os.path.join(
                pool_dir, split_name, inst["channel"], inst["source"] + ".npy"
            )
        )
    return _POOL_CACHE[key]


def _instance_seed(inst):
    """由样本 id 生成固定种子（用于验证集可复现）"""
    return int(hashlib.md5(inst["id"].encode()).hexdigest()[:8], 16)


def _make_sample(inst, split_name):
    """单个样本：截取 -> 加噪 -> 归一化 -> STFT(RI)"""
    wave = _load_pool_for(inst, split_name)
    length = inst["length"]
    if split_name == "train":
        # 每次取样本都重新随机截取起点（在线增广）
        start = int(np.random.randint(0, len(wave) - length + 1))
    else:
        # 验证集固定起点，保证各 epoch 可比
        start = inst["start"]
    clean = np.asarray(wave[start:start + length], dtype=np.float64)

    if split_name == "train":
        # 每次取样本都重新随机加噪（在线增广）
        noise, _ = generate_mt_noise(length, FS, seed=None)
        snr = float(np.random.uniform(*SNR_RANGE))
    else:
        # 验证集固定噪声与 SNR，保证各 epoch 可比
        noise, _ = generate_mt_noise(length, FS, seed=_instance_seed(inst))
        snr = CV_SNR

    noisy = mix_noisy(clean, noise, snr)

    # 用干净信号的能量归一化（对 noisy 和 clean 用同一个系数）
    c = np.sqrt(length / (np.sum(clean ** 2) + EPSILON))
    clean = clean * c
    noisy = noisy * c

    clean_t = torch.FloatTensor(clean)
    noisy_t = torch.FloatTensor(noisy)
    label = mt_stft(clean_t, N_FFT, HOP, WINDOW)     # [1, 2, T, F]
    feat = mt_stft(noisy_t, N_FFT, HOP, WINDOW)
    return feat[0], label[0]                          # [2, T, F]


def generate_feats_labels(minibatch, split_name):
    """一个 minibatch（长度一致的样本列表）-> 特征与标签"""
    feat_list, label_list = [], []
    for inst in minibatch:
        feat, label = _make_sample(inst, split_name)
        feat_list.append(feat)
        label_list.append(label)
    return (
        torch.stack(feat_list, dim=0),
        torch.stack(label_list, dim=0),
        None,
    )


class TrainDataset(Dataset):
    """返回 minibatch 列表（每个 minibatch 内长度一致）"""

    def __init__(self, json_dir, batch_size):
        with open(os.path.join(json_dir, "train", "files.json"), "r") as f:
            minibatch_list = json.load(f)
        self.minibatch = minibatch_list

    def __len__(self):
        return len(self.minibatch)

    def __getitem__(self, index):
        return self.minibatch[index]


class CvDataset(Dataset):

    def __init__(self, json_dir, batch_size):
        with open(os.path.join(json_dir, "cv", "files.json"), "r") as f:
            minibatch_list = json.load(f)
        self.minibatch = minibatch_list

    def __len__(self):
        return len(self.minibatch)

    def __getitem__(self, index):
        return self.minibatch[index]


class BatchInfo(object):
    def __init__(self, feats, labels, frame_mask_list=None):
        self.feats = feats
        self.labels = labels
        self.frame_mask_list = frame_mask_list


class TrainDataLoader(object):
    """shuffle=True：每个 epoch 随机打乱 minibatch 顺序"""

    def __init__(self, data_set, **kw):
        self.data_loader = DataLoader(
            dataset=data_set,
            shuffle=True,
            batch_size=1,
            collate_fn=self.collate_fn,
            **kw,
        )

    @staticmethod
    def collate_fn(batch):
        feats, labels, _ = generate_feats_labels(batch[0], "train")
        return BatchInfo(feats, labels)

    def get_data_loader(self):
        return self.data_loader


class CvDataLoader(object):

    def __init__(self, data_set, **kw):
        self.data_loader = DataLoader(
            dataset=data_set,
            shuffle=False,
            batch_size=1,
            collate_fn=self.collate_fn,
            **kw,
        )

    @staticmethod
    def collate_fn(batch):
        feats, labels, _ = generate_feats_labels(batch[0], "cv")
        return BatchInfo(feats, labels)

    def get_data_loader(self):
        return self.data_loader


if __name__ == "__main__":
    # 快速自检：数据加载 + STFT 形状
    ds = TrainDataset(json_dir, batch_size)
    mb = ds[0]
    feats, labels, _ = generate_feats_labels(mb, "train")
    print("minibatch 样本数:", len(mb))
    print("feats:", tuple(feats.shape), "labels:", tuple(labels.shape))
