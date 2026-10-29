# -*- coding: utf-8 -*-
"""
MT 推理：整段 STFT -> 网络 -> iSTFT（与语音 DBT-Net 的整句推理一致）

输入是一条连续 MT 波形（任意长度），不切块：
    waveform -> STFT [2, 257, T] -> MT-AIA-Net -> [2, 257, T] -> iSTFT

输出（按通道目录）：
    <out>/<channel>/<stem>_noisy.npy         带噪波形（原始输入）
    <out>/<channel>/<stem>_denoised.npy      去噪后波形
    <out>/<channel>/<stem>_noisy_ri.npy      带噪复数谱 RI [2, F, T]
    <out>/<channel>/<stem>_denoised_ri.npy   去噪复数谱 RI [2, F, T]（可直接用于阻抗估计）

如果提供 --clean，会同时打印去噪前后的 SNR。
--channels 可一次处理多个通道（各通道独立去噪）：
    python infer_mt.py --channels ex,ey,hx,hy --stem ModEM7
"""
import argparse
import os
import numpy as np
import torch

from config_mt import (
    FS,
    N_FFT,
    HOP,
    WINDOW,
    model_best_path,
)
from mt_stft import mt_stft, mt_istft
from mt_aia_net import MTDualAiaTransMergeCRM


def snr_db(clean, est):
    clean = np.asarray(clean, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)
    noise = est - clean
    return 10.0 * np.log10(
        (np.sum(clean ** 2) + 1e-12) / (np.sum(noise ** 2) + 1e-12)
    )


def infer_one(input_path, channel, args):
    """处理单个通道的一段连续波形"""
    device = torch.device(args.device if torch.cuda.is_available()
                          else "cpu")
    model = MTDualAiaTransMergeCRM(freq_bins=N_FFT // 2 + 1)
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    wave = np.load(input_path).astype(np.float64)
    if args.samples is not None:
        wave = wave[:args.samples]
    length = len(wave)

    # 与训练一致的归一化（用输入自身能量）
    c = np.sqrt(length / (np.sum(wave ** 2) + 1e-12))
    wave_norm = wave * c

    with torch.no_grad():
        feat = mt_stft(
            torch.FloatTensor(wave_norm).unsqueeze(0), N_FFT, HOP, WINDOW
        ).to(device)                                   # [1, 2, T, F]
        esti = model(feat)                              # [1, 2, T, F]
        est_wave = mt_istft(
            esti.cpu(), length=length, n_fft=N_FFT, hop=HOP, window=WINDOW
        ).squeeze(0).numpy()

    denoised = est_wave / c

    stem = args.stem or os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.join(args.output, channel)
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, f"{stem}_noisy.npy"), wave)
    np.save(os.path.join(out_dir, f"{stem}_denoised.npy"), denoised)

    # 保存复数谱 RI [2, F, T]，供 mt_impedance.py 使用
    def save_ri(feat_tensor, name):
        ri = feat_tensor[0].permute(0, 2, 1).contiguous().numpy()  # [2, F, T]
        np.save(os.path.join(out_dir, f"{stem}_{name}.npy"), ri)

    save_ri(feat.cpu(), "noisy_ri")
    save_ri(esti.cpu(), "denoised_ri")

    print(f"通道: {channel}, 输入: {input_path}")
    print(f"长度: {length} 点, 时间帧 T = {feat.shape[2]}, F = {feat.shape[3]}")
    print(f"输出目录: {out_dir}")

    if args.clean is not None:
        clean = np.load(args.clean).astype(np.float64)[:length]
        snr_in = snr_db(clean, wave)
        snr_out = snr_db(clean, denoised)
        print(f"输入 SNR: {snr_in:.2f} dB, 去噪后 SNR: {snr_out:.2f} dB, "
              f"提升: {snr_out - snr_in:.2f} dB")


def infer(args):
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    for channel in channels:
        input_path = args.input.format(channel=channel)
        if not os.path.isfile(input_path):
            print(f"[跳过] 找不到输入: {input_path}")
            continue
        infer_one(input_path, channel, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MT inference")
    parser.add_argument("--checkpoint", type=str, default=model_best_path)
    parser.add_argument("--input", type=str,
                        default="./split/{channel}/ModEM7.npy",
                        help="输入波形模板，{channel} 会被替换（任意长度，整段处理）")
    parser.add_argument("--channels", type=str, default="ex",
                        help="逗号分隔的通道列表，如 ex 或 ex,ey,hx,hy")
    parser.add_argument("--stem", type=str, default=None,
                        help="输出文件名主干（默认取输入文件名）")
    parser.add_argument("--output", type=str, default="./MT_enhanced")
    parser.add_argument("--samples", type=int, default=None,
                        help="只取前 N 个采样点（不填则整段）")
    parser.add_argument("--clean", type=str, default=None,
                        help="干净参考波形（可选，用于计算 SNR 提升）")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    infer(args)
