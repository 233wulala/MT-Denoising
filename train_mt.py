# -*- coding: utf-8 -*-
"""
MT 训练入口

流程（与语音 DBT-Net 一致）：
    1. python split_channels.py    # 按通道拆分 STS 文件（可选，已有输出可跳过）
    2. python prepare_mt_data.py   # 按时间划分 train/cv + 变长随机截取样本清单
    3. python train_mt.py          # 训练（在线加噪 + STFT + 随机打乱）

运行示例：
    python train_mt.py
    python train_mt.py --epochs 200 --batch_size 8 --lr 5e-5
"""
import argparse
import os
import sys
import numpy as np
import torch

from config_mt import (
    json_dir,
    loss_dir,
    model_best_path,
    check_point_path,
    batch_size,
    epochs,
    lr,
)
from data_mt import TrainDataset, CvDataset, TrainDataLoader, CvDataLoader
from mt_aia_net import MTDualAiaTransMergeCRM
from solver_mt import SolverMT
from Backup import numParams


def main(args, model):
    # 强制 stdout 逐行刷新：nohup/重定向/远程运行时也能实时看到打印
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    tr_dataset = TrainDataset(json_dir=args.json_dir, batch_size=args.batch_size)
    cv_dataset = CvDataset(json_dir=args.json_dir, batch_size=args.cv_batch_size)

    tr_loader = TrainDataLoader(
        data_set=tr_dataset,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    cv_loader = CvDataLoader(
        data_set=cv_dataset,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    data = {"tr_loader": tr_loader, "cv_loader": cv_loader}

    print(model)
    print(
        "The number of trainable parameters of the net is:%d"
        % numParams(model)
    )

    optimizer = torch.optim.Adam(
        model.parameters(), args.lr, weight_decay=args.l2
    )
    solver = SolverMT(data, model, optimizer, args)
    solver.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MT-AIA-DBT-Net")
    parser.add_argument("--json_dir", type=str, default=json_dir)
    parser.add_argument("--loss_dir", type=str, default=loss_dir)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--cv_batch_size", type=int, default=batch_size)
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--early_stop", dest="early_stop", default=1, type=int)
    parser.add_argument("--half_lr", type=int, default=1)
    parser.add_argument("--shuffle", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--l2", type=float, default=1e-7)
    parser.add_argument("--best_path", default=model_best_path)
    parser.add_argument("--cp_path", type=str, default=check_point_path)
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--is_conti", type=bool, default=False)
    parser.add_argument("--conti_path", type=str, default="")
    args = parser.parse_args()

    model = MTDualAiaTransMergeCRM(freq_bins=257)
    main(args, model)
