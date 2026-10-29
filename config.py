# s2 根据您的目录更改参数设置
import os

# front-end parameter settings
chunk_length = 128
feat_type = 'log'   # normal, sqrt, cubic, log_1x
is_conti = False
conti_path = './CP_dir/checkpoint_early_exit_16th.pth.tar'
is_pesq = False

# server parameter settings
json_dir = 'Json/train/files.json'
file_path = 'meas'
loss_dir = './LOSS/aia_internew_vb.mat'
batch_size = 4
epochs = 100
lr = 1e-4
model_best_path = './BEST_MODEL/aia_internew_vb.pth.tar'
check_point_path = './CP_dir/aia_internew_vb'

os.makedirs('./BEST_MODEL', exist_ok=True)
os.makedirs('./LOSS', exist_ok=True)
os.makedirs(check_point_path, exist_ok=True)