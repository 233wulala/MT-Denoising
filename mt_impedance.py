# -*- coding: utf-8 -*-
"""
MT 阻抗估计：直接在频域从（去噪后的）复数谱计算阻抗张量

思路（迁移自 DBT-Net 讨论结论）：
    网络输出的是复数谱 X̂(f,t)，不需要转回时域，
    直接跨时间帧做统计估计得到阻抗：

        E(f,t) = [Ex(f,t), Ey(f,t)]^T
        H(f,t) = [Hx(f,t), Hy(f,t)]^T
        Z(f)   = <E H^H> <H H^H>^{-1}      （2x2 复矩阵，逐频率）

    再由 Z 计算视电阻率与相位：
        rho_a(f) = |Z(f)|^2 / (omega * mu0)
        phi(f)   = arg(Z(f))

输入：四个通道的复数谱 RI npy，形状 [2, F, T]（实部/虚部, 频率, 时间帧），
     由 infer_mt.py 保存（*_ri.npy）。

用法：
    python mt_impedance.py \
        --ex MT_enhanced/ex/ModEM7_denoised_ri.npy \
        --ey MT_enhanced/ey/ModEM7_denoised_ri.npy \
        --hx MT_enhanced/hx/ModEM7_denoised_ri.npy \
        --hy MT_enhanced/hy/ModEM7_denoised_ri.npy \
        --out MT_enhanced/impedance_denoised
"""
import argparse
import os
import numpy as np

from config_mt import FS, N_FFT

MU0 = 4.0 * np.pi * 1e-7      # 真空磁导率 (H/m)
REG = 1e-6                    # 对角加载系数（防止矩阵奇异）


def load_ri(path):
    """读取 [2, F, T] 的 RI 谱并转为复数谱 [F, T]"""
    ri = np.load(path)         # [2, F, T]
    return ri[0] + 1j * ri[1]  # [F, T]


def compute_impedance(ri_ex, ri_ey, ri_hx, ri_hy):
    """
    输入：4 个 [F, T] 复数谱（Ex, Ey, Hx, Hy）
    返回：
        Z      [2, 2, F] 复数阻抗张量（Z[0,0]=Zxx, Z[0,1]=Zxy, Z[1,0]=Zyx, Z[1,1]=Zyy）
        rho_a  [2, 2, F] 视电阻率 (Ohm*m)
        phase  [2, 2, F] 相位 (度)
        freqs  [F]
    """
    ex, ey, hx, hy = ri_ex, ri_ey, ri_hx, ri_hy
    f, t = ex.shape
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / FS)   # 0 ~ 7.5 Hz
    if len(freqs) != f:
        raise ValueError(
            f"频率维 {f} 与 n_fft={N_FFT} 得到的 {len(freqs)} 不一致"
        )

    E = np.stack([ex, ey], axis=0)             # [2, F, T]
    H = np.stack([hx, hy], axis=0)             # [2, F, T]

    Z = np.zeros((2, 2, f), dtype=np.complex128)
    for k in range(f):
        Ek = E[:, k, :]                        # [2, T]
        Hk = H[:, k, :]                        # [2, T]
        s_eh = Ek @ Hk.conj().T                # [2, 2]
        s_hh = Hk @ Hk.conj().T                # [2, 2]
        s_hh = s_hh + REG * np.eye(2) * np.trace(s_hh).real
        Z[:, :, k] = s_eh @ np.linalg.inv(s_hh)

    omega = 2.0 * np.pi * freqs
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_a = np.abs(Z) ** 2 / (omega * MU0)[None, None, :]
        phase = np.angle(Z, deg=True)

    rho_a[..., 0] = np.nan      # f=0 处无定义
    phase[..., 0] = np.nan
    return Z, rho_a, phase, freqs


def save_results(out_prefix, Z, rho_a, phase, freqs):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    np.save(out_prefix + "_Z.npy", Z)
    np.save(out_prefix + "_rhoa.npy", rho_a)
    np.save(out_prefix + "_phase.npy", phase)
    np.save(out_prefix + "_freqs.npy", freqs)

    # 常用分量 CSV：Zxy / Zyx 的视电阻率与相位
    csv_path = out_prefix + ".csv"
    comp = {"Zxx": (0, 0), "Zxy": (0, 1), "Zyx": (1, 0), "Zyy": (1, 1)}
    with open(csv_path, "w") as f:
        header = "freq_Hz," + ",".join(
            f"rhoa_{k}_Ohmm,phase_{k}_deg" for k in comp
        )
        f.write(header + "\n")
        for k in range(len(freqs)):
            row = [f"{freqs[k]:.6f}"]
            for (i, j) in comp.values():
                row.append(f"{rho_a[i, j, k]:.6E}")
                row.append(f"{phase[i, j, k]:.6f}")
            f.write(",".join(row) + "\n")
    print(f"已保存: {out_prefix}_Z.npy / _rhoa.npy / _phase.npy / {csv_path}")


def main():
    parser = argparse.ArgumentParser("MT impedance estimation")
    parser.add_argument("--ex", required=True)
    parser.add_argument("--ey", required=True)
    parser.add_argument("--hx", required=True)
    parser.add_argument("--hy", required=True)
    parser.add_argument("--out", required=True, help="输出前缀，例如 ./result/impedance")
    args = parser.parse_args()

    ri_ex = load_ri(args.ex)
    ri_ey = load_ri(args.ey)
    ri_hx = load_ri(args.hx)
    ri_hy = load_ri(args.hy)

    Z, rho_a, phase, freqs = compute_impedance(
        ri_ex, ri_ey, ri_hx, ri_hy
    )

    print("Zxy 前 5 个频点:")
    for k in range(min(5, len(freqs))):
        print(
            f"  f={freqs[k]:.4f} Hz  Zxy={Z[0,1,k]:.4e} "
            f"rhoa={rho_a[0,1,k]:.4e} Ohm*m  phase={phase[0,1,k]:.2f} deg"
        )
    save_results(args.out, Z, rho_a, phase, freqs)


if __name__ == "__main__":
    main()
