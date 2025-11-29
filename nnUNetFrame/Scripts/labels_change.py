#标签映射
import os
import SimpleITK as sitk
import numpy as np

# 定义标签映射
label_mapping = {
    205: 1,
    420: 2,
    500: 3,
    550: 4,
    600: 5,
    820: 6,
    850: 7
}

# 标签文件所在目录
label_dir = r'C:\DeepLearning\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\Dataset002_WHS++CTTrain\labelsTr'

# 遍历目录中的所有标签文件
for filename in os.listdir(label_dir):
    if filename.endswith('.nii.gz'):
        # 构建文件路径
        file_path = os.path.join(label_dir, filename)

        # 读取标签文件
        label_img = sitk.ReadImage(file_path)
        label_data = sitk.GetArrayFromImage(label_img)

        # 替换标签
        for original, new in label_mapping.items():
            label_data[label_data == original] = new

        # 保存修改后的标签文件（保持原文件名）
        new_label_img = sitk.GetImageFromArray(label_data)
        sitk.WriteImage(new_label_img, file_path)

        print(f"Processed: {filename}")

print("标签修改完成。")
