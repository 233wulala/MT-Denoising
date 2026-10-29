# -*- coding: utf-8 -*-
"""
MT 版 AIA-DBT 网络（迁移自语音 DBT-Net 的 dual_aia_trans_merge_crm）

核心设计：
    - 频率维度固定：输入 [B, 2, T, F=257]（RI 两通道，单 MT 通道）
      编码器 257 -> 128（频率方向 stride=2 下采样）
      解码器 128 -> 256 -> 257（亚像素卷积 + 补 1 点）
    - 时间维度 T 可变：所有时间方向操作（膨胀卷积、双路径注意力）
      都不依赖固定 T，任意帧数均可前向
    - 双分支：幅值分支 + RI 分支，经 AIA_Transformer_merge 交互

如果修改 n_fft，只需保证 FREQ_BINS 为奇数（n_fft 为 2 的幂时成立）。
"""
import torch
import torch.nn as nn

from aia_inter_new import AIA_Transformer_merge, AHAM
from Backup import numParams


class MTDenseEncoder(nn.Module):
    """RI 分支编码器：[B, 2, T, F] -> [B, 64, T, F//2]"""

    def __init__(self, freq_bins=257, in_channels=2, width=64):
        super(MTDenseEncoder, self).__init__()
        self.freq_bins = freq_bins
        self.mid_bins = freq_bins // 2
        self.width = width
        self.inp_conv = nn.Conv2d(in_channels, width, kernel_size=(1, 1))
        self.inp_norm = nn.LayerNorm(freq_bins)
        self.inp_prelu = nn.PReLU(width)
        self.enc_dense1 = DenseBlockMT(freq_bins, 4, width)
        self.enc_conv1 = nn.Conv2d(
            width, width, kernel_size=(1, 3), stride=(1, 2)
        )
        self.enc_norm1 = nn.LayerNorm(self.mid_bins)
        self.enc_prelu1 = nn.PReLU(width)

    def forward(self, x):
        out = self.inp_prelu(self.inp_norm(self.inp_conv(x)))
        out = self.enc_dense1(out)
        x = self.enc_prelu1(self.enc_norm1(self.enc_conv1(out)))
        return x


class MTDenseEncoderMag(nn.Module):
    """幅值分支编码器：[B, 1, T, F] -> [B, 64, T, F//2]"""

    def __init__(self, freq_bins=257, in_channels=1, width=64):
        super(MTDenseEncoderMag, self).__init__()
        self.freq_bins = freq_bins
        self.mid_bins = freq_bins // 2
        self.width = width
        self.inp_conv = nn.Conv2d(in_channels, width, kernel_size=(1, 1))
        self.inp_norm = nn.LayerNorm(freq_bins)
        self.inp_prelu = nn.PReLU(width)
        self.enc_dense1 = DenseBlockMT(freq_bins, 4, width)
        self.enc_conv1 = nn.Conv2d(
            width, width, kernel_size=(1, 3), stride=(1, 2)
        )
        self.enc_norm1 = nn.LayerNorm(self.mid_bins)
        self.enc_prelu1 = nn.PReLU(width)

    def forward(self, x):
        out = self.inp_prelu(self.inp_norm(self.inp_conv(x)))
        out = self.enc_dense1(out)
        x = self.enc_prelu1(self.enc_norm1(self.enc_conv1(out)))
        return x


class MTDenseDecoder(nn.Module):
    """解码器：[B, 64, T, F//2] -> [B, 1, T, F]"""

    def __init__(self, freq_bins=257, width=64):
        super(MTDenseDecoder, self).__init__()
        self.freq_bins = freq_bins
        self.mid_bins = freq_bins // 2
        self.pad = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.pad1 = nn.ConstantPad2d((1, 0, 0, 0), value=0.0)
        self.width = width
        self.dec_dense1 = DenseBlockMT(self.mid_bins, 4, width)
        self.dec_conv1 = SPConvTranspose2d(
            width, width, kernel_size=(1, 3), r=2
        )
        self.dec_norm1 = nn.LayerNorm(freq_bins)
        self.dec_prelu1 = nn.PReLU(width)
        self.out_conv = nn.Conv2d(width, 1, kernel_size=(1, 1))

    def forward(self, x):
        out = self.dec_dense1(x)
        out = self.dec_prelu1(
            self.dec_norm1(self.pad1(self.dec_conv1(self.pad(out))))
        )
        out = self.out_conv(out)
        return out


class MTDenseDecoderMasking(nn.Module):
    """幅值掩码解码器：[B, 64, T, F//2] -> [B, 1, T, F]（0~1）"""

    def __init__(self, freq_bins=257, width=64):
        super(MTDenseDecoderMasking, self).__init__()
        self.freq_bins = freq_bins
        self.mid_bins = freq_bins // 2
        self.pad = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.pad1 = nn.ConstantPad2d((1, 0, 0, 0), value=0.0)
        self.width = width
        self.dec_dense1 = DenseBlockMT(self.mid_bins, 4, width)
        self.dec_conv1 = SPConvTranspose2d(
            width, width, kernel_size=(1, 3), r=2
        )
        self.dec_norm1 = nn.LayerNorm(freq_bins)
        self.dec_prelu1 = nn.PReLU(width)
        self.mask1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, 1)),
            nn.Sigmoid(),
        )
        self.mask2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, 1)),
            nn.Tanh(),
        )
        self.maskconv = nn.Conv2d(1, 1, kernel_size=(1, 1))
        self.maskrelu = nn.Sigmoid()
        self.out_conv = nn.Conv2d(width, 1, kernel_size=(1, 1))

    def forward(self, x):
        out = self.dec_dense1(x)
        out = self.dec_prelu1(
            self.dec_norm1(self.pad1(self.dec_conv1(self.pad(out))))
        )
        out = self.out_conv(out)
        out = self.mask1(out) * self.mask2(out)
        out = self.maskrelu(self.maskconv(out))
        return out


class MTDualAiaTransMergeCRM(nn.Module):
    """
    MT 版双分支网络（幅值 + RI）
    输入 [B, 2, T, F=257]，输出 [B, 2, T, F=257]
    T 任意。
    """

    def __init__(self, freq_bins=257):
        super(MTDualAiaTransMergeCRM, self).__init__()
        self.freq_bins = freq_bins
        self.en_ri = MTDenseEncoder(freq_bins, in_channels=2)
        self.en_mag = MTDenseEncoderMag(freq_bins, in_channels=1)
        self.aia_trans_merge = AIA_Transformer_merge(
            128, 64, num_layers=4, norm_size=freq_bins // 2
        )
        self.aham = AHAM(input_channel=64)
        self.aham_mag = AHAM(input_channel=64)

        self.de1 = MTDenseDecoder(freq_bins)
        self.de2 = MTDenseDecoder(freq_bins)
        self.de_mag_mask = MTDenseDecoderMasking(freq_bins)

    def forward(self, x):
        batch_size, _, seq_len, _ = x.shape
        x_mag_ori, x_phase_ori = torch.norm(x, dim=1), torch.atan2(
            x[:, -1, :, :], x[:, 0, :, :]
        )
        x_mag = x_mag_ori.unsqueeze(dim=1)

        # 双分支编码 + 交互
        x_ri = self.en_ri(x)
        x_mag_en = self.en_mag(x_mag)
        _, x_outputlist_mag, _, x_outputlist_ri = self.aia_trans_merge(
            x_mag_en, x_ri
        )

        x_ri = self.aham(x_outputlist_ri)
        x_mag_en = self.aham_mag(x_outputlist_mag)
        x_mag_mask = self.de_mag_mask(x_mag_en)
        x_mag_mask = x_mag_mask.squeeze(dim=1)

        # RI 解码
        x_real = self.de1(x_ri).squeeze(dim=1)
        x_imag = self.de2(x_ri).squeeze(dim=1)

        # 幅值掩码 + RI 残差合并
        x_mag_out = x_mag_mask * x_mag_ori
        x_r_out = x_mag_out * torch.cos(x_phase_ori) + x_real
        x_i_out = x_mag_out * torch.sin(x_phase_ori) + x_imag
        x_com_out = torch.stack((x_r_out, x_i_out), dim=1)
        return x_com_out


class SPConvTranspose2d(nn.Module):
    """亚像素转置卷积：只沿频率（最后）维度上采样 r 倍"""

    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels * r, kernel_size=kernel_size,
            stride=(1, 1),
        )
        self.r = r

    def forward(self, x):
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class DenseBlockMT(nn.Module):
    """膨胀密集块（时间方向因果，频率方向保持尺寸不变）"""

    def __init__(self, input_size, depth=5, in_channels=64):
        super(DenseBlockMT, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(
                self, "pad{}".format(i + 1),
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
            )
            setattr(
                self, "conv{}".format(i + 1),
                nn.Conv2d(
                    self.in_channels * (i + 1), self.in_channels,
                    kernel_size=self.kernel_size, dilation=(dil, 1),
                ),
            )
            setattr(self, "norm{}".format(i + 1), nn.LayerNorm(input_size))
            setattr(
                self, "prelu{}".format(i + 1), nn.PReLU(self.in_channels)
            )

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, "pad{}".format(i + 1))(skip)
            out = getattr(self, "conv{}".format(i + 1))(out)
            out = getattr(self, "norm{}".format(i + 1))(out)
            out = getattr(self, "prelu{}".format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out


if __name__ == "__main__":
    model = MTDualAiaTransMergeCRM(freq_bins=257)
    model.eval()
    for t in (15, 31, 63):
        x = torch.FloatTensor(2, 2, t, 257)
        y = model(x)
        print(f"输入 [2, 2, {t}, 257] -> 输出 {tuple(y.shape)}")
    print("可训练参数:", numParams(model))
