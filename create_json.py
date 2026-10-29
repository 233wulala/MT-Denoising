import os
import json

def create_mt_json(data_dir, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    file_list = []

    files = sorted(os.listdir(data_dir))

    for file in files:

        if file.endswith(".npy"):

            file_name = os.path.splitext(file)[0]

            file_list.append(file_name)

    json_file = os.path.join(
        save_dir,
        "files.json"
    )

    with open(json_file, "w") as f:

        json.dump(
            file_list,
            f,
            indent=4
        )

    print(f"Total Samples: {len(file_list)}")
    print(f"Saved: {json_file}")

# ============================
# 使用
# ============================

create_mt_json(
    data_dir="MT_tf",
    save_dir="MT_dataset/Json/train"
)