"""
This script is the backup function used to support backup support for the SE system
Author: Andong Li
Time: 2019/05
"""
import torch
import torch.nn as nn
import librosa
import pickle
import json
import os
#import h5py
import numpy as np
#from pesq import pesq
from joblib import Parallel, delayed
from scipy import signal
import sys
from functools import reduce
from torch.nn.modules.module import _addindent
import scipy.linalg as linalg
# from config import feat_type, fft_num, win_size, win_shift
EPSILON = 1e-10



def numParams(net):
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num


def mt_mse_loss(
    esti,
    label
):

    return torch.mean(
        (esti-label)**2
    )


def mt_complex_mag_loss(esti, label, frame_list=None, w_complex=1.0, w_mag=0.5):
    """
    MT 损失：复谱误差 + 幅值误差（带可选 frame mask）

        L_complex = 1/N * sum( (Re_hat - Re)^2 + (Im_hat - Im)^2 )
        L_mag     = 1/N * sum( (|X_hat| - |X|)^2 )
        L         = w_complex * L_complex + w_mag * L_mag

    输入输出形状 [B, 2, T, F]，通道维 0=实部, 1=虚部。
    frame_list：每句有效帧数（变长 batch 用）；None 表示全部帧有效。

    权重说明：
        w_complex=1.0, w_mag=0.5 -> L_complex + 0.5*L_mag
        w_complex=0.5, w_mag=0.5 -> 与原版 DBT-Net com_mag_mse_loss
                                    (0.5*(L1+L2)) 一致
    """
    x_r_hat = esti[:, 0]
    x_i_hat = esti[:, 1]
    x_r = label[:, 0]
    x_i = label[:, 1]

    mag_hat = torch.sqrt(x_r_hat ** 2 + x_i_hat ** 2 + EPSILON)
    mag = torch.sqrt(x_r ** 2 + x_i ** 2 + EPSILON)

    if frame_list is None:
        # 等长 batch：全帧有效
        # 对全部 [B, 2, T, F] 元素求平均（N = 2*B*T*F），与原版
        # com_mag_mse_loss 的 sum/com_mask.sum() 归一化一致
        l_complex = torch.mean((esti - label) ** 2)
        l_mag = torch.mean((mag_hat - mag) ** 2)
        return w_complex * l_complex + w_mag * l_mag

    # 变长 batch：按帧数生成 mask（与 DBT-Net 原版一致）
    b, _, t, f = esti.shape
    mask = torch.zeros((b, t, f), dtype=esti.dtype, device=esti.device)
    for i, n in enumerate(frame_list):
        mask[i, :n] = 1.0
    com_mask = torch.stack((mask, mask), dim=1)     # [B, 2, T, F]

    l_complex = (
        ((esti - label) * com_mask) ** 2
    ).sum() / com_mask.sum()
    l_mag = (
        ((mag_hat - mag) * mask) ** 2
    ).sum() / mask.sum()
    return w_complex * l_complex + w_mag * l_mag
