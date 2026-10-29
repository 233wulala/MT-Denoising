# -*- coding: utf-8 -*-
"""
MT 数据准备（按源划分 train / cv）

    1. 从 split/<channel>/<source>.npy 读取按通道拆分好的连续波形
    2. 按源划分：TRAIN_SOURCES（默认 6 个）整源进入训练池，
       CV_SOURCES（默认 2 个）整源进入验证池
       —— 验证集来自训练中完全未见过的源（泛化测试，对应
          “Model A 训练 / Model B 测试”的实验设计）
    3. 保存干净数据池：MT_dataset/pool/{train,cv}/<channel>/<source>.npy
    4. 生成样本清单（files.json）：
        训练样本只记录 (通道, 源, 长度)，起点由 data_mt.py 每次随机截取；
        验证样本记录固定起点，保证各 epoch 可比。

训练样本长度在 TRAIN_LENGTHS 中随机选择（4096~16384 点，对应 T≈17~65），
验证样本固定为 CV_LENGTH。每个 minibatch 内的样本长度一致，方便直接堆叠。
"""
import json
import os
import numpy as np

from config_mt import (
    SPLIT_DIR,
    pool_dir,
    json_dir,
    TRAIN_LENGTHS,
    CV_LENGTH,
    N_TRAIN_SAMPLES,
    N_CV_SAMPLES,
    TRAIN_SOURCES,
    CV_SOURCES,
    TARGET_CHANNELS,
    batch_size,
)


# ==========================================
# 1. 按源划分并保存数据池
# ==========================================


def build_pools():
    os.makedirs(os.path.join(pool_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(pool_dir, "cv"), exist_ok=True)

    all_sources = set()
    for channel in TARGET_CHANNELS:
        ch_dir = os.path.join(SPLIT_DIR, channel)
        if not os.path.isdir(ch_dir):
            print(f"[跳过] 通道目录不存在: {ch_dir}")
            continue
        sources = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(ch_dir) if f.endswith(".npy")
        )
        all_sources.update(sources)

        for src in sources:
            if src in TRAIN_SOURCES:
                split_name = "train"
            elif src in CV_SOURCES:
                split_name = "cv"
            else:
                print(f"[跳过] {src} 不在 TRAIN_SOURCES / CV_SOURCES 中")
                continue

            wave = np.load(os.path.join(ch_dir, src + ".npy"))
            dest_dir = os.path.join(pool_dir, split_name, channel)
            os.makedirs(dest_dir, exist_ok=True)
            np.save(os.path.join(dest_dir, src + ".npy"), wave)
            print(f"[pool] {channel}/{src} -> {split_name} ({len(wave)} 点)")

    # 校验划分是否覆盖全部源
    missing = sorted(all_sources - set(TRAIN_SOURCES) - set(CV_SOURCES))
    if missing:
        print(f"[警告] 以下源未分配 train/cv: {missing}")
    print(
        f"[划分] 训练源 {len(TRAIN_SOURCES)} 个, 验证源 {len(CV_SOURCES)} 个, "
        f"共 {len(all_sources)} 个源"
    )
    return all_sources


# ==========================================
# 2. 生成样本清单（按 minibatch 分组）
# ==========================================


def make_instances(pool_root, split_name, lengths, counts, with_start):
    """
    生成样本实例并分成 minibatch（每个 minibatch 内长度一致）。
    with_start=True（cv）时记录固定起点；False（train）时不记录，
    由 data_mt.py 每次随机截取。
    """
    rng = np.random.default_rng(2026)
    instances = []

    # 预读各源文件长度
    ch_dir = os.path.join(pool_root, split_name)
    channels = sorted(os.listdir(ch_dir))
    src_len = {}
    for channel in channels:
        for f in sorted(os.listdir(os.path.join(ch_dir, channel))):
            if f.endswith(".npy"):
                src = os.path.splitext(f)[0]
                src_len[(channel, src)] = int(
                    np.load(os.path.join(ch_dir, channel, f)).shape[0]
                )

    # 每个长度平均分配样本数，并取 batch_size 的整数倍
    count_per_len = counts // len(lengths)
    count_per_len = max(batch_size, count_per_len // batch_size * batch_size)

    global_id = 0
    for length in lengths:
        # 只保留长度足够的源，避免死循环
        candidates = [
            (c, s) for (c, s), ln in src_len.items() if ln >= length
        ]
        if not candidates:
            raise ValueError(
                f"{split_name} 池中没有长度 >= {length} 的数据，"
                "请检查数据或调小 TRAIN_LENGTHS"
            )
        created = 0
        while created < count_per_len:
            channel, src = candidates[int(rng.integers(len(candidates)))]
            wave_len = src_len[(channel, src)]
            start = int(rng.integers(0, wave_len - length + 1))
            inst = {
                "id": f"{split_name}_{channel}_{src}_{length}_{global_id}",
                "channel": channel,
                "source": src,
                "length": length,
            }
            if with_start:
                inst["start"] = start
            instances.append(inst)
            created += 1
            global_id += 1

    # 按顺序切成 minibatch（同一长度连续生成 -> 每批长度一致）
    minibatches = [
        instances[i:i + batch_size]
        for i in range(0, len(instances), batch_size)
    ]
    print(f"[{split_name}] 样本数: {len(instances)}, minibatch 数: {len(minibatches)}")
    return minibatches


def save_json():
    os.makedirs(os.path.join(json_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(json_dir, "cv"), exist_ok=True)

    train_mb = make_instances(
        pool_dir, "train", TRAIN_LENGTHS, N_TRAIN_SAMPLES, with_start=False
    )
    cv_mb = make_instances(
        pool_dir, "cv", [CV_LENGTH], N_CV_SAMPLES, with_start=True
    )

    with open(os.path.join(json_dir, "train", "files.json"), "w") as f:
        json.dump(train_mb, f, indent=2)
    with open(os.path.join(json_dir, "cv", "files.json"), "w") as f:
        json.dump(cv_mb, f, indent=2)

    print(f"[json] {os.path.join(json_dir, 'train', 'files.json')}")
    print(f"[json] {os.path.join(json_dir, 'cv', 'files.json')}")


if __name__ == "__main__":
    build_pools()
    save_json()
    print("数据准备完成。下一步运行 python train_mt.py 开始训练。")
