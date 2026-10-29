# -*- coding: utf-8 -*-
"""
为测试集批量生成带噪文件：

    读取 split/<channel>/<source>.npy（干净波形）
    -> 按固定窗长切段（不重叠，覆盖整段数据）
    -> 每段用与训练相同的 mt_noise 噪声模型在时域加噪（固定种子，可复现）
    -> 输出成对的 noisy / clean 文件，供 infer_mt.py 去噪评估

输出目录结构（默认 ./MT_test）：
    MT_test/
      noisy/<channel>/<source>_<编号>.npy
      clean/<channel>/<source>_<编号>.npy
      <source>_<channel>_manifest.csv   （记录每段的起点/长度/SNR/种子）

用法：
    python make_noisy_test.py --source ModEM8 --channel ex --snr 5

评估示例（对每个测试文件跑推理并看 SNR 提升）：
    python infer_mt.py --input MT_test/noisy/ex/ModEM8_0000.npy \
        --clean MT_test/clean/ex/ModEM8_0000.npy --channel ex \
        --stem ModEM8_0000
"""
import argparse
import os
import csv
import numpy as np

from config_mt import FS, CV_SNR, SPLIT_DIR
from mt_noise import generate_mt_noise, mix_noisy


def main():
    parser = argparse.ArgumentParser("MT 测试集批量加噪")
    parser.add_argument("--source", type=str, default="ModEM8",
                        help="源文件名（不带 .npy）")
    parser.add_argument("--channel", type=str, default="ex")
    parser.add_argument("--out_dir", type=str, default="./MT_test")
    parser.add_argument("--length", type=int, default=16384,
                        help="每段长度（对应 T≈65）")
    parser.add_argument("--snr", type=float, default=CV_SNR,
                        help="加噪 SNR (dB)，默认取验证集 CV_SNR")
    parser.add_argument("--seed", type=int, default=2026,
                        help="噪声种子基准，每段 seed = seed + 编号")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="最多生成几段（默认全部不重叠段）")
    args = parser.parse_args()

    input_path = os.path.join(SPLIT_DIR, args.channel, args.source + ".npy")
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"找不到文件: {input_path}")
    clean_wave = np.load(input_path).astype(np.float64)

    length = args.length
    total = len(clean_wave)
    n_full = total // length
    n_samples = n_full if args.n_samples is None else min(args.n_samples, n_full)
    if n_samples == 0:
        raise ValueError(f"数据长度 {total} 不足一个 {length} 点的段")

    noisy_dir = os.path.join(args.out_dir, "noisy", args.channel)
    clean_dir = os.path.join(args.out_dir, "clean", args.channel)
    os.makedirs(noisy_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)

    manifest = os.path.join(
        args.out_dir, f"{args.source}_{args.channel}_manifest.csv"
    )
    with open(manifest, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "start", "length", "snr_db", "seed"])

        for idx in range(n_samples):
            start = idx * length
            clean = clean_wave[start:start + length]
            seed = args.seed + idx
            noise, _ = generate_mt_noise(length, FS, seed=seed)
            noisy = mix_noisy(clean, noise, args.snr)

            name = f"{args.source}_{idx:04d}"
            np.save(os.path.join(noisy_dir, name + ".npy"), noisy)
            np.save(os.path.join(clean_dir, name + ".npy"), clean)
            writer.writerow([name, start, length, f"{args.snr:.2f}", seed])

            # 抽查第一个样本的实际 SNR，确认与设定一致
            if idx == 0:
                measured = 10.0 * np.log10(
                    (np.sum(clean ** 2) + 1e-12)
                    / (np.sum((noisy - clean) ** 2) + 1e-12)
                )
                print(f"检查：设定 SNR={args.snr:.2f} dB，实测 {measured:.2f} dB")

    print(f"输入: {input_path} ({total} 点)")
    print(f"段长: {length}（丢弃尾部 {total - n_samples * length} 点）")
    print(f"生成: {n_samples} 个测试样本")
    print(f"噪声目录: {noisy_dir}")
    print(f"干净目录: {clean_dir}")
    print(f"清单: {manifest}")


if __name__ == "__main__":
    main()
