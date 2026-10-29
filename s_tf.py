"""
数据预处理2：批量生成时频图样本
"""
import os
import cv2
import numpy as np
from stockwell import st

input_dir = "MT_segments"
output_dir = "MT_tf"

os.makedirs(output_dir, exist_ok=True)

files = sorted(os.listdir(input_dir))

for file in files:

    data = np.load(
        os.path.join(input_dir, file)
    )

    Ex = data[0]
    Ey = data[1]
    Hx = data[2]
    Hy = data[3]

    # ------------------
    # S Transform
    # ------------------

    S_ex = st.st(Ex)
    S_ey = st.st(Ey)

    S_hx = st.st(Hx)
    S_hy = st.st(Hy)

    # ------------------
    # Cross Power
    # ------------------

    C_exhy = S_ex * np.conj(S_hy)

    C_eyhx = S_ey * np.conj(S_hx)

    X = np.stack([
        np.real(C_exhy),
        np.imag(C_exhy),

        np.real(C_eyhx),
        np.imag(C_eyhx)
    ], axis=0)

    # 数据归一化
    X = X / np.max(
        np.abs(X)
    )
    # ------------------
    # log压缩
    # ------------------

    def normalize(x):
        x = np.sign(x) * np.log1p(np.abs(x))

        return x

    R1 = normalize(
        np.real(C_exhy)
    )

    I1 = normalize(
        np.imag(C_exhy)
    )

    R2 = normalize(
        np.real(C_eyhx)
    )

    I2 = normalize(
        np.imag(C_eyhx)
    )

    # ------------------
    # resize
    # ------------------

    C1 = cv2.resize(R1, (128,128))
    C2 = cv2.resize(I1, (128,128))

    C3 = cv2.resize(R2, (128,128))
    C4 = cv2.resize(I2, (128,128))

    X = np.stack([
        R1,
        I1,
        R2,
        I2
    ], axis=0)

    save_path = file.replace(
        ".npy",
        "_tf.npy"
    )

    np.save(
        os.path.join(output_dir, save_path),
        X.astype(np.float32)
    )

print("Finished.")