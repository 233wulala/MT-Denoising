"""
数据预处理1:切窗并保存为 npy 文件
"""
import os
import numpy as np

# ============================
# 参数
# ============================

window = 4096
stride = 2048

save_dir = "MT_segments"

os.makedirs(save_dir, exist_ok=True)

# ============================
# 读取数据
# 这里替换成你的数据
# ============================

Ex = np.loadtxt("meas/ex.ascii")
Ey = np.loadtxt("meas/ey.ascii")
Hx = np.loadtxt("meas/bx.ascii")
Hy = np.loadtxt("meas/by.ascii")

# ============================
# 检查长度
# ============================

N = len(Ex)

assert len(Ey) == N
assert len(Hx) == N
assert len(Hy) == N

# ============================
# 切窗
# ============================

sample_id = 0

for start in range(0, N - window + 1, stride):

    end = start + window

    sample = np.stack(
        [
            Ex[start:end],
            Ey[start:end],
            Hx[start:end],
            Hy[start:end]
        ],
        axis=0
    )

    save_name = f"sample_{sample_id:05d}.npy"

    np.save(
        os.path.join(save_dir, save_name),
        sample
    )

    sample_id += 1

print("Total samples:", sample_id)