# -*- coding: utf-8 -*-
"""
对推理（去噪）结果做质量评价，与干净参考对比。

指标定义（全部在时域波形上计算）：
    SNR    = 10*log10( sum(ref^2) / sum((est-ref)^2) )        dB
    MSE    = mean( (est-ref)^2 )
    RMSE   = sqrt(MSE)
    NRMSE  = RMSE / RMS(ref) = sqrt( sum((est-ref)^2) / sum(ref^2) )
    MAE    = mean( |est-ref| )
    CORC   = Pearson 相关系数 corr(ref, est)                    [-1, 1]

输入目录约定：
    enhanced_dir/<channel>/<stem>_denoised.npy   （infer_mt.py 的输出）
    enhanced_dir/<channel>/<stem>_noisy.npy      （带噪输入，推理时一并保存，作为基线）
    clean_dir/<channel>/<stem>.npy               （干净参考，如 make_noisy_test.py 的输出）

用法：
    python evaluate_mt.py --channel ex \
        --clean_dir ./MT_test/clean --enhanced_dir ./MT_enhanced

输出：
    终端打印每个文件的指标与平均值；
    同时保存 CSV：<enhanced_dir>/evaluation_<channel>.csv
"""
import argparse
import os
import csv
import numpy as np


def metrics(ref, est):
    """返回 dict：snr/mse/rmse/nrmse/mae/corc"""
    ref = np.asarray(ref, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)
    n = min(len(ref), len(est))
    ref = ref[:n]
    est = est[:n]
    if n == 0:
        raise ValueError("空序列，无法计算指标")

    err = est - ref
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    ref_rms = float(np.sqrt(np.mean(ref ** 2)))
    ref_energy = float(np.sum(ref ** 2))
    err_energy = float(np.sum(err ** 2))
    mae = float(np.mean(np.abs(err)))

    if err_energy > 1e-30 and ref_energy > 1e-30:
        snr = 10.0 * np.log10(ref_energy / err_energy)
    else:
        snr = float("inf") if err_energy <= 1e-30 else float("-inf")

    nrmse = rmse / ref_rms if ref_rms > 1e-30 else float("nan")

    if ref.std() > 1e-30 and est.std() > 1e-30:
        corc = float(np.corrcoef(ref, est)[0, 1])
    else:
        corc = float("nan")

    return {
        "snr": snr, "mse": mse, "rmse": rmse,
        "nrmse": nrmse, "mae": mae, "corc": corc,
    }


def fmt_row(m):
    return (
        f"SNR={m['snr']:8.2f} dB  "
        f"MSE={m['mse']:.6e}  "
        f"RMSE={m['rmse']:.6e}  "
        f"NRMSE={m['nrmse']:.6f}  "
        f"MAE={m['mae']:.6e}  "
        f"CORC={m['corc']:.6f}"
    )


def mean_metrics(list_of_m):
    """对每个指标求均值（跳过 inf/nan 的指标值）"""
    keys = ["snr", "mse", "rmse", "nrmse", "mae", "corc"]
    out = {}
    for k in keys:
        vals = [m[k] for m in list_of_m if np.isfinite(m[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def main():
    parser = argparse.ArgumentParser("MT 推理质量评价")
    parser.add_argument("--channel", type=str, default="ex")
    parser.add_argument("--clean_dir", type=str, default="./MT_test/clean")
    parser.add_argument("--enhanced_dir", type=str, default="./MT_enhanced",
                        help="infer_mt.py 的输出目录")
    parser.add_argument("--out", type=str, default=None,
                        help="CSV 输出路径（默认 <enhanced_dir>/evaluation_<channel>.csv）")
    args = parser.parse_args()

    enhanced_ch = os.path.join(args.enhanced_dir, args.channel)
    clean_ch = os.path.join(args.clean_dir, args.channel)
    if not os.path.isdir(enhanced_ch):
        raise FileNotFoundError(f"找不到推理输出目录: {enhanced_ch}")
    if not os.path.isdir(clean_ch):
        raise FileNotFoundError(f"找不到干净参考目录: {clean_ch}")

    # 找所有 *_denoised.npy
    denoised_files = sorted(
        f for f in os.listdir(enhanced_ch) if f.endswith("_denoised.npy")
    )
    if not denoised_files:
        raise FileNotFoundError(
            f"{enhanced_ch} 下没有 *_denoised.npy，请先运行 infer_mt.py"
        )

    rows = []
    for df in denoised_files:
        stem = df[: -len("_denoised.npy")]
        clean_path = os.path.join(clean_ch, stem + ".npy")
        noisy_path = os.path.join(enhanced_ch, stem + "_noisy.npy")
        if not os.path.isfile(clean_path):
            print(f"[跳过] 缺少干净参考: {clean_path}")
            continue

        ref = np.load(clean_path)
        est = np.load(os.path.join(enhanced_ch, df))
        m_denoised = metrics(ref, est)

        m_noisy = None
        if os.path.isfile(noisy_path):
            m_noisy = metrics(ref, np.load(noisy_path))

        rows.append((stem, m_noisy, m_denoised))
        print(f"[{stem}]")
        if m_noisy:
            print(f"  输入(带噪): {fmt_row(m_noisy)}")
        print(f"  去噪后   : {fmt_row(m_denoised)}")

    if not rows:
        raise ValueError("没有可评价的样本")

    avg_noisy = mean_metrics([m for _, m, _ in rows if m is not None])
    avg_denoised = mean_metrics([m for _, _, m in rows])
    print("-" * 100)
    print(f"平均（{len(rows)} 个样本）:")
    if rows[0][1] is not None:
        print(f"  输入(带噪): {fmt_row(avg_noisy)}")
    print(f"  去噪后   : {fmt_row(avg_denoised)}")

    out_path = args.out or os.path.join(
        args.enhanced_dir, f"evaluation_{args.channel}.csv"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "file", "snr_noisy_db", "mse_noisy", "nrmse_noisy", "mae_noisy",
            "corc_noisy", "snr_denoised_db", "mse_denoised", "nrmse_denoised",
            "mae_denoised", "corc_denoised",
        ])
        for stem, m_noisy, m_denoised in rows:
            def cell(m, k):
                return "" if m is None else f"{m[k]:.8f}"

            w.writerow([
                stem,
                cell(m_noisy, "snr"), cell(m_noisy, "mse"),
                cell(m_noisy, "nrmse"), cell(m_noisy, "mae"),
                cell(m_noisy, "corc"),
                cell(m_denoised, "snr"), cell(m_denoised, "mse"),
                cell(m_denoised, "nrmse"), cell(m_denoised, "mae"),
                cell(m_denoised, "corc"),
            ])
        w.writerow([
            "MEAN",
            f"{avg_noisy['snr']:.8f}", f"{avg_noisy['mse']:.8f}",
            f"{avg_noisy['nrmse']:.8f}", f"{avg_noisy['mae']:.8f}",
            f"{avg_noisy['corc']:.8f}",
            f"{avg_denoised['snr']:.8f}", f"{avg_denoised['mse']:.8f}",
            f"{avg_denoised['nrmse']:.8f}", f"{avg_denoised['mae']:.8f}",
            f"{avg_denoised['corc']:.8f}",
        ])
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
