# -*- coding: utf-8 -*-
"""
MT 训练求解器：训练/验证循环、学习率衰减、早停、断点保存。
损失：L = w_complex * L_complex + w_mag * L_mag
    （复谱实部/虚部 MSE + 幅值 MSE，权重见 config_mt.LOSS_W_COMPLEX / LOSS_W_MAG）
"""
import os
import time
import gc
import numpy as np
import torch

from Backup import mt_complex_mag_loss
from config_mt import LOSS_W_COMPLEX, LOSS_W_MAG


class SolverMT(object):

    def __init__(self, data, model, optimizer, args):
        self.tr_loader = data["tr_loader"]
        self.cv_loader = data["cv_loader"]
        self.model = model
        self.optimizer = optimizer

        self.loss_dir = args.loss_dir
        self.epochs = args.epochs
        self.half_lr = args.half_lr
        self.early_stop = args.early_stop
        self.best_path = args.best_path
        self.cp_path = args.cp_path
        self.print_freq = args.print_freq
        self.is_conti = args.is_conti
        self.conti_path = args.conti_path

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"[solver] device = {self.device}")
        self.model.to(self.device)

        self.tr_epoch = []
        self.cv_epoch = []
        self._reset()

    def _reset(self):
        if self.is_conti:
            checkpoint = torch.load(
                self.conti_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["epoch"]
            self.prev_cv_loss = checkpoint["cv_loss"]
            self.best_cv_loss = checkpoint["best_cv_loss"]
            self.cv_no_impv = 0
            self.having = False
        else:
            self.start_epoch = 0
            self.prev_cv_loss = float("inf")
            self.best_cv_loss = float("inf")
            self.cv_no_impv = 0
            self.having = False

    def _save_loss(self):
        os.makedirs(os.path.dirname(self.loss_dir), exist_ok=True)
        np.save(self.loss_dir + ".npy", np.array([self.tr_epoch, self.cv_epoch]))
        with open(self.loss_dir + ".txt", "w") as f:
            f.write("epoch\ttrain_loss\tcv_loss\n")
            for i, (t, c) in enumerate(zip(self.tr_epoch, self.cv_epoch), 1):
                f.write(f"{i}\t{t:.6f}\t{c:.6f}\n")

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            print("Begin to train.....")
            self.model.train()
            start = time.time()
            tr_avg_loss = self.run_one_epoch(epoch, cross_valid=False)
            print("-" * 90)
            print(
                "End of Epoch %d, Time: %4f s, Train_Loss:%5f"
                % (int(epoch + 1), time.time() - start, tr_avg_loss)
            )
            print("-" * 90)

            # Cross validation
            print("Begin Cross Validation....")
            self.model.eval()
            cv_avg_loss = self.run_one_epoch(epoch, cross_valid=True)
            print("-" * 90)
            print("Time: %4fs, CV_Loss:%5f" % (time.time() - start, cv_avg_loss))
            print("-" * 90)

            # save checkpoint
            check_point = {
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "tr_loss": tr_avg_loss,
                "cv_loss": cv_avg_loss,
                "best_cv_loss": self.best_cv_loss,
            }
            torch.save(
                check_point,
                os.path.join(
                    self.cp_path, "checkpoint_early_exit_%dth.pth.tar"
                    % (epoch + 1)
                ),
            )

            self.tr_epoch.append(tr_avg_loss)
            self.cv_epoch.append(cv_avg_loss)
            self._save_loss()

            # Adjust learning rate and early stop
            if self.half_lr:
                if cv_avg_loss >= self.prev_cv_loss:
                    self.cv_no_impv += 1
                    if self.cv_no_impv == 3:
                        self.having = True
                    if self.cv_no_impv >= 5 and self.early_stop:
                        print("No improvement and apply early stop")
                        break
                else:
                    self.cv_no_impv = 0

            if self.having:
                optim_state = self.optimizer.state_dict()
                for i in range(len(optim_state["param_groups"])):
                    optim_state["param_groups"][i]["lr"] /= 2.0
                self.optimizer.load_state_dict(optim_state)
                print(
                    "Learning rate adjusted to %5f"
                    % (optim_state["param_groups"][0]["lr"])
                )
                self.having = False
            self.prev_cv_loss = cv_avg_loss

            if cv_avg_loss < self.best_cv_loss:
                self.best_cv_loss = cv_avg_loss
                torch.save(self.model.state_dict(), self.best_path)
                print(
                    "Find better cv model, saving to %s"
                    % os.path.split(self.best_path)[1]
                )

    def run_one_epoch(self, epoch, cross_valid=False):
        start1 = time.time()
        total_loss = 0.0
        data_loader = (
            self.tr_loader if not cross_valid else self.cv_loader
        )
        batch_id = 0
        for batch_id, batch_info in enumerate(data_loader.get_data_loader()):
            batch_feat = batch_info.feats.to(self.device)
            batch_label = batch_info.labels.to(self.device)

            esti = self.model(batch_feat)
            loss = mt_complex_mag_loss(
                esti,
                batch_label,
                frame_list=None,
                w_complex=LOSS_W_COMPLEX,
                w_mag=LOSS_W_MAG,
            )

            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            gc.collect()
            if batch_id % self.print_freq == 0:
                print(
                    "Epoch:%d, Iter:%d, the average_loss:%5f, "
                    "current_loss:%5f, %d ms/batch."
                    % (
                        int(epoch + 1), int(batch_id),
                        total_loss / (batch_id + 1), loss.item(),
                        1000 * (time.time() - start1) / (batch_id + 1),
                    )
                )
        return total_loss / (batch_id + 1)
