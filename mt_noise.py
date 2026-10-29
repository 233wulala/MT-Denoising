# -*- coding: utf-8 -*-
"""
合成 MT 时域噪声模型

在时域生成接近真实 MT 观测噪声的噪声序列，包括：
    - 有色噪声（1/f^alpha，随机 alpha）
    - 若干正弦谐波（文化噪声 / 周期干扰）
    - 随机尖峰脉冲
    - 慢漂移（基线）
    - 白噪声底（传感器噪声）

用法：先 generate_mt_noise 得到噪声波形，再 mix_noisy 按 SNR 叠加到干净信号上。
加噪发生在时域，STFT 在其之后——与语音 DBT-Net 的顺序一致。
"""
import numpy as np


def _colored_noise(rng, length, alpha):
    """1/f^alpha 有色噪声（FFT 频谱整形）"""
    white = rng.standard_normal(length)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(length)
    freqs[0] = 1.0                      # 保护直流分量
    spec = spec / (freqs ** (alpha / 2.0))
    spec[0] = 0.0
    return np.fft.irfft(spec, length)


def _harmonics(rng, length, fs, n_harm, fmin, fmax):
    """若干随机频率、随机相位的正弦谐波叠加"""
    t = np.arange(length) / fs
    signal = np.zeros(length)
    nyq = fs / 2.0
    fmax = min(fmax, nyq * 0.95)
    for _ in range(n_harm):
        freq = rng.uniform(fmin, fmax)
        amp = rng.uniform(0.2, 1.5)
        phase = rng.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * freq * t + phase)
    return signal


def _spikes(rng, length, n_spikes):
    """随机尖峰脉冲"""
    signal = np.zeros(length)
    positions = rng.integers(0, length, size=n_spikes)
    for pos in positions:
        amp = rng.choice([-1.0, 1.0]) * rng.uniform(2.0, 8.0)
        width = int(rng.integers(1, 3))
        signal[pos:pos + width] += amp
    return signal


def _drift(rng, length):
    """慢漂移：随机游走后再平滑"""
    walk = np.cumsum(rng.standard_normal(length))
    # 简单滑动平均（等价于低通）
    window = max(1, length // 200)
    kernel = np.ones(window) / window
    smooth = np.convolve(walk, kernel, mode="same")
    smooth -= smooth.mean()
    return smooth * 0.05 / (np.abs(smooth).max() + 1e-12)


def generate_mt_noise(length, fs, seed=None):
    """
    生成一段合成 MT 噪声 [length]。

    返回 (noise, components)：
        noise     - 组合后的噪声波形（幅度量级 ~1，叠加时按 SNR 缩放）
        components- 各成分字典，便于调试/调权
    """
    rng = np.random.default_rng(seed)

    alpha = float(rng.uniform(*NOISE_COLORED_ALPHA))
    colored = _colored_noise(rng, length, alpha)
    colored = colored / (np.abs(colored).max() + 1e-12)

    n_harm = int(rng.integers(*NOISE_HARMONICS))
    harm = _harmonics(rng, length, fs, n_harm,
                      NOISE_HARMONIC_FREQ[0], NOISE_HARMONIC_FREQ[1])
    harm = harm / (np.abs(harm).max() + 1e-12)

    n_spikes = int(rng.integers(*NOISE_SPIKES))
    spikes = _spikes(rng, length, n_spikes)
    spikes = spikes / (np.abs(spikes).max() + 1e-12)

    drift = _drift(rng, length) if NOISE_DRIFT else np.zeros(length)
    sensor = rng.standard_normal(length) / 3.0 if NOISE_SENSOR else np.zeros(length)

    noise = (
        1.0 * colored
        + 0.35 * harm
        + 0.15 * spikes
        + 0.30 * drift
        + 0.05 * sensor
    )
    components = {
        "colored": colored,
        "harmonics": harm,
        "spikes": spikes,
        "drift": drift,
        "sensor": sensor,
    }
    return noise, components


def mix_noisy(clean, noise, snr_db):
    """
    按目标 SNR 把噪声叠加到干净信号上（时域混合）：
        noisy = clean + noise * gain
    SNR 定义为 10*log10(sum(clean^2) / sum((gain*noise)^2))
    """
    clean = np.asarray(clean, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    clean_power = np.sum(clean ** 2) + 1e-12
    noise_power = np.sum(noise ** 2) + 1e-12
    gain = np.sqrt(clean_power / noise_power) * 10 ** (-snr_db / 20.0)
    return clean + gain * noise


# 运行时从 config_mt 注入噪声参数，避免循环导入
def _load_noise_params():
    from config_mt import (
        NOISE_COLORED_ALPHA,
        NOISE_HARMONICS,
        NOISE_HARMONIC_FREQ,
        NOISE_SPIKES,
        NOISE_DRIFT,
        NOISE_SENSOR,
    )
    globals().update(
        NOISE_COLORED_ALPHA=NOISE_COLORED_ALPHA,
        NOISE_HARMONICS=NOISE_HARMONICS,
        NOISE_HARMONIC_FREQ=NOISE_HARMONIC_FREQ,
        NOISE_SPIKES=NOISE_SPIKES,
        NOISE_DRIFT=NOISE_DRIFT,
        NOISE_SENSOR=NOISE_SENSOR,
    )


_load_noise_params()
