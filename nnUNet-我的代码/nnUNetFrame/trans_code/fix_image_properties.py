import SimpleITK as sitk
import os
from pathlib import Path


def fix_label_properties(image_path, label_path):
    # 读取图像和标签文件
    image = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path)

    # 设置标签文件的属性与图像文件一致
    label.SetSpacing(image.GetSpacing())
    label.SetOrigin(image.GetOrigin())
    label.SetDirection(image.GetDirection())

    # 保存修正后的标签文件
    sitk.WriteImage(label, label_path)


def process_dataset(images_dir, labels_dir):
    # 获取所有图像文件
    image_files = [f for f in os.listdir(images_dir) if f.endswith('_0000.nii.gz')]

    for image_file in image_files:
        # 获取对应的标签文件名
        label_file = image_file.replace('_0000.nii.gz', '.nii.gz')

        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, label_file)

        # 修正标签文件的属性
        fix_label_properties(image_path, label_path)


if __name__ == '__main__':
    images_dir = r'C:\DeepLearning\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\Dataset002_WHS++CTTrain\imagesTr'
    labels_dir = r'C:\DeepLearning\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\Dataset002_WHS++CTTrain\labelsTr'

    process_dataset(images_dir, labels_dir)
