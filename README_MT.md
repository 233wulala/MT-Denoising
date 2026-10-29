# MT 版 DBT-Net（时频去噪 + 阻抗估计）

把语音 DBT-Net 的四个核心思想迁移到 MT 数据：

1. **频率维度固定**：`F = n_fft/2 + 1 = 257`（n_fft=512, hop=256, Hann 窗），
   网络真正固定的只有频率维。
2. **时间维度可变**：训练随机截取 4096~16384 点（T≈17~65），
   推理时整段 STFT → 网络 → iSTFT，不切块、不拼接。
3. **时域加噪**：噪声在波形域按随机 SNR 叠加，STFT 在其之后。
4. **随机打乱**：训练时每个 epoch 的批次顺序随机，且每次取样本
   都重新随机截取、随机加噪（在线增广）。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `split_channels.py` | STS 文件按 ex/ey/hx/hy/hz 分通道输出 |
| `prepare_mt_data.py` | 按源划分 6 train / 2 cv，生成变长截取样本清单 |
| `data_mt.py` | 数据加载：变长截取 + 时域加噪 + STFT |
| `mt_stft.py` | STFT/iSTFT（RI 表示，F=257，T 可变） |
| `mt_noise.py` | 合成 MT 噪声（有色噪声/谐波/尖峰/漂移/白噪底） |
| `mt_aia_net.py` | MT 版双分支网络（幅值 + RI），输入 `[B,2,T,257]` |
| `solver_mt.py` / `train_mt.py` | 训练循环与入口 |
| `infer_mt.py` | 整段推理：STFT → 网络 → iSTFT，输出波形和复数谱 |
| `mt_impedance.py` | 由复数谱直接估计阻抗张量 → 视电阻率、相位 |
| `config_mt.py` | 所有参数集中配置 |

## 运行流程

```bash
# 1. 分通道（dataset/*.STS* -> split/<channel>/*.npy）
python split_channels.py dataset/ModEM0.STS1

# 2. 数据准备（按源划分：6 个 train / 2 个 cv + 变长样本清单）
python prepare_mt_data.py

# 3. 训练（在线加噪 + STFT）
python train_mt.py --epochs 100 --batch_size 4

# 4. 整段推理（可一次处理多个通道）
python infer_mt.py --channels ex,ey,hx,hy --stem ModEM7 --samples 16384

# 5. 阻抗估计（直接使用去噪后的复数谱，无需回到时域）
python mt_impedance.py \
    --ex MT_enhanced/ex/ModEM7_denoised_ri.npy \
    --ey MT_enhanced/ey/ModEM7_denoised_ri.npy \
    --hx MT_enhanced/hx/ModEM7_denoised_ri.npy \
    --hy MT_enhanced/hy/ModEM7_denoised_ri.npy \
    --out MT_enhanced/impedance_ModEM7
```

## 常用参数（config_mt.py）

- `FS`：采样率，默认 15 Hz（与 STS 文件实际采样率一致）
- `TRAIN_LENGTHS`：训练截取长度，对应 T=17/33/49/65
- `SNR_RANGE` / `CV_SNR`：训练随机 SNR 范围、验证固定 SNR
- `TRAIN_SOURCES` / `CV_SOURCES`：按源划分 train/cv（默认 6 / 2）
- `TARGET_CHANNELS`：参与数据准备的通道（单通道实验先设 `["ex"]`）
- `N_FFT / HOP / WINDOW`：STFT 参数，修改后 `FREQ_BINS` 自动跟随
  （注意 F 需为奇数，n_fft 取 2 的幂即可）
- `LOSS_W_COMPLEX` / `LOSS_W_MAG`：损失权重，默认 1.0 / 0.5；
  改成 0.5 / 0.5 即复现原版 DBT-Net 的 `com_mag_mse_loss`

## 与语音 DBT-Net 的对应关系

| 语音 | MT 版 |
| --- | --- |
| 161 频点（n_fft=320） | 257 频点（n_fft=512） |
| 固定 3 秒训练截取 | 随机 4.5~18 分钟（4096~16384 点） |
| 整句推理 | 整段推理，T 可变 |
| 时域加噪后 STFT | 同上 |
| iSTFT 输出语音 | iSTFT 输出波形；也可直接用复数谱算阻抗 |
