# -*- coding: utf-8 -*-
"""
将 dataset 中的 STS 时间序列文件按通道拆分为
ex / ey / hx / hy / hz 五个目录，每个目录下存放
以数据来源命名的独立文件。

输入文件每行格式（文本）：
    时间戳  值1  值2  值3  值4  值5
其中 5 个数值分别对应 ex, ey, hx, hy, hz 五个通道。
"""

import os
import numpy as np

# ============================================================
# 路径配置：直接修改下面的变量即可，改完保存后运行本文件
# ============================================================

# 输入：要拆分的 STS 文件名（放在 DATASET_DIR 目录下）
# 也可以填完整路径，例如 "D:/data/xxx.STS1"
INPUT_FILE = "ModEM8.STS3"

# 数据所在目录（INPUT_FILE 为文件名时，从这里找）
DATASET_DIR = "dataset"

# 输出根目录，其下会自动创建 ex / ey / hx / hy / hz 子目录
OUTPUT_DIR = "split"

# 输出格式："npy" / "txt" / "csv" / "all"（all = npy + txt）
OUTPUT_FORMAT = "npy"

# ============================================================
# 以下代码一般无需修改
# ============================================================

CHANNELS = ["ex", "ey", "hx", "hy", "hz"]


def split_channels(input_path, output_dir, fmt):
    """读取 STS 文件，按通道拆分并保存。"""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"找不到文件: {input_path}")

    base = os.path.basename(input_path)
    stem = os.path.splitext(base)[0]
    os.makedirs(output_dir, exist_ok=True)

    # 逐行读取并收集各通道数据，避免一次把整个文件读进内存
    channel_data = [[] for _ in CHANNELS]
    line_count = 0

    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # 第 0 列是时间戳，第 1~5 列是 5 个通道
            if len(parts) != 1 + len(CHANNELS):
                raise ValueError(
                    f"第 {line_count + 1} 行格式错误: 期望 {1 + len(CHANNELS)} 列, "
                    f"实际 {len(parts)} 列"
                )
            for i, col in enumerate(parts[1:], start=0):
                channel_data[i].append(float(col))
            line_count += 1

    if line_count == 0:
        raise ValueError("文件为空，没有任何数据")

    # 按通道分目录保存
    saved = []
    for name, values in zip(CHANNELS, channel_data):
        arr = np.asarray(values, dtype=np.float64)
        channel_dir = os.path.join(output_dir, name)
        os.makedirs(channel_dir, exist_ok=True)
        if fmt in ("npy", "all"):
            npy_path = os.path.join(channel_dir, f"{stem}.npy")
            np.save(npy_path, arr)
            saved.append(npy_path)
        if fmt in ("txt", "csv", "all"):
            suffix = ".csv" if fmt == "csv" else ".txt"
            txt_path = os.path.join(channel_dir, f"{stem}{suffix}")
            np.savetxt(txt_path, arr, fmt="%.10E")
            saved.append(txt_path)

    print(f"输入文件: {input_path}")
    print(f"数据行数: {line_count}")
    print(f"输出根目录: {output_dir}")
    for name, values in zip(CHANNELS, channel_data):
        arr = np.asarray(values)
        print(f"  {name}: 长度={arr.size}, 均值={arr.mean():.6E}, "
              f"最小值={arr.min():.6E}, 最大值={arr.max():.6E}")
    print("已保存:")
    for path in saved:
        print(f"  {path}")


def main():
    # 解析 INPUT_FILE：文件名 -> 拼上 DATASET_DIR；完整路径 -> 直接用
    if os.path.isfile(INPUT_FILE):
        input_path = INPUT_FILE
    else:
        input_path = os.path.join(DATASET_DIR, INPUT_FILE)

    split_channels(input_path, OUTPUT_DIR, OUTPUT_FORMAT)


if __name__ == "__main__":
    main()
