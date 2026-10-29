# -*- coding: utf-8 -*-
"""
MT 版 STFT / iSTFT

约定：
    - n_fft=512, hop=256, Hann 窗, center=True（与语音 DBT-Net 的 torch.stft 一致）
    - 频率维度 F = n_fft/2 + 1 = 257（固定）
    - 时间维度 T 随波形长度变化：
        T = L // hop + 1
      例：L=4096 -> T=17, L=8192 -> T=33, L=16384 -> T=65
      （与设计目标 T≈15~63 同一量级，频率维 F 才是网络真正固定的维度）
    - 网络输入输出为 RI 表示 [B, 2, T, F]（通道维 = [实部, 虚部]）
"""
import torch


def get_window(n_fft, window="hann"):
    if window == "hann":
        return torch.hann_window(n_fft)
    raise ValueError(f"不支持的窗函数: {window}")


def stft_to_ri(complex_spec):
    """复数谱 [B, F, T] -> RI [B, 2, T, F]"""
    real = complex_spec.real
    imag = complex_spec.imag
    ri = torch.stack([real, imag], dim=1)          # [B, 2, F, T]
    return ri.permute(0, 1, 3, 2).contiguous()     # [B, 2, T, F]


def ri_to_stft(ri):
    """RI [B, 2, T, F] -> 复数谱 [B, F, T]"""
    ri = ri.permute(0, 1, 3, 2).contiguous()       # [B, 2, F, T]
    return torch.complex(ri[:, 0], ri[:, 1])


def mt_stft(waveform, n_fft=512, hop=256, window="hann"):
    """
    波形 [B, L] -> RI 特征 [B, 2, T, F]
    波形可以是一维 [L] 或二维 [B, L]。
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    win = get_window(n_fft, window)
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win,
        center=True,
        return_complex=True,
    )                                          # [B, F, T]
    return stft_to_ri(spec)


def mt_istft(ri, length=None, n_fft=512, hop=256, window="hann"):
    """
    RI 特征 [B, 2, T, F] -> 波形 [B, L]
    """
    spec = ri_to_stft(ri)
    win = get_window(n_fft, window)
    return torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win,
        center=True,
        length=length,
    )


def frame_count(length, n_fft=512, hop=256):
    """给定波形长度 L，返回 STFT 时间帧数 T"""
    if length <= 0:
        return 0
    return length // hop + 1
