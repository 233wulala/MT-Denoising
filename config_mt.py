# -*- coding: utf-8 -*-
"""
MT 版 DBT-Net 配置

核心思想（迁移自语音 DBT-Net）：
    1. 频率维度 F 固定：F = n_fft/2 + 1 = 257
    2. 时间维度 T 可变：训练时随机截取不同长度，推理时整段处理
    3. 加噪发生在时域（波形），STFT 在加噪之后
    4. 训练样本随机打乱:
"""
import os

# ==========================
# 数据与采样
# ==========================

FS = 15.0                     # 采样率 (Hz)

# 当前做单通道实验，从 ["ex","ey","hx","hy","hz"] 中选
CHANNELS = ["ex", "ey", "hx", "hy", "hz"]
TARGET_CHANNELS = ["ex"]      # 要参与本次数据准备的通道

# 分通道数据目录（split_channels.py 的输出：split/ex/xxx.npy 等）
SPLIT_DIR = "./split"

# ==========================
# STFT 参数（固定 F=257）
# ==========================

N_FFT = 512
WIN_SIZE = 512
HOP = 256
WINDOW = "hann"
FREQ_BINS = N_FFT // 2 + 1     # 257

# ==========================
# 训练长度（时间维度可变）
# ==========================

# 4096 -> T=17, 8192 -> T=33, 12288 -> T=49, 16384 -> T=65
# （与设计目标 T≈15~63 同一量级；频率维 F=257 才是网络固定的维度）
TRAIN_LENGTHS = [4096, 8192, 12288, 16384]
CV_LENGTH = 8192               # 验证集固定长度

N_TRAIN_SAMPLES = 5000         # 生成的训练样本条数
N_CV_SAMPLES = 500             # 生成的验证样本条数

# ==========================
# 按源划分 train / cv（8 个源 -> 6 train / 2 cv）
# ==========================

TRAIN_SOURCES = [
    "ModEM0", "ModEM1", "ModEM2",
    "ModEM3", "ModEM4", "ModEM5",
]
CV_SOURCES = ["ModEM6", "ModEM7"]

# ==========================
# 损失函数权重
# L = w_complex * L_complex + w_mag * L_mag
# ==========================

# 当前公式：L_complex + 0.5 * L_mag
LOSS_W_COMPLEX = 1.0
LOSS_W_MAG = 0.5
# 复现原版 DBT-Net com_mag_mse_loss（0.5*(L1+L2)）时改为：
# LOSS_W_COMPLEX = 0.5
# LOSS_W_MAG = 0.5

# ==========================
# 噪声模型（时域合成 MT 噪声，随机 SNR）
# ==========================

SNR_RANGE = [-5.0, 10.0]       # 随机 SNR (dB) 范围
CV_SNR = 5.0                   # 验证集固定 SNR (dB)

# 各类噪声成分的权重（1 = 保留该成分）
NOISE_COLORED_ALPHA = [0.5, 2.0]   # 有色噪声 1/f^alpha 的 alpha 范围
NOISE_HARMONICS = [1, 4]           # 谐波数量范围
NOISE_HARMONIC_FREQ = [0.02, 2.5]  # 谐波频率范围 (Hz)，不超过 Nyquist
NOISE_SPIKES = [5, 30]             # 尖峰数量范围
NOISE_DRIFT = True                 # 是否加慢漂移
NOISE_SENSOR = True                # 是否加白噪声底

# ==========================
# 路径
# ==========================

json_dir = "./MT_dataset/Json"
file_path = "./MT_dataset"
pool_dir = "./MT_dataset/pool"

batch_size = 4
epochs = 100
lr = 1e-4

loss_dir = "./LOSS/mt_loss"
model_best_path = "./BEST_MODEL/mt_best.pth.tar"
check_point_path = "./CP_dir/mt"

os.makedirs("./BEST_MODEL", exist_ok=True)
os.makedirs("./LOSS", exist_ok=True)
os.makedirs(check_point_path, exist_ok=True)
