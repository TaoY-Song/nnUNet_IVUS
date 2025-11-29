import os
import shutil
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import save_json, join


def generate_dataset_json(output_folder: str, channel_names: dict, labels: dict, num_training_cases: int,
                          file_ending: str, dataset_name: str, overwrite_image_reader_writer: str):
    dataset_json = {
        'channel_names': channel_names,
        'labels': labels,
        'numTraining': num_training_cases,
        'file_ending': file_ending,
        'name': dataset_name,
        'overwrite_image_reader_writer': overwrite_image_reader_writer
    }
    save_json(dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)


def convert_dataset(source_dir: str, base_dir: Path, dataset_name: str):
    # 创建输出目录
    out_dir = base_dir / dataset_name
    imagesTr = out_dir / 'imagesTr'
    labelsTr = out_dir / 'labelsTr'
    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)

    # 定义图像和标签的命名模式
    image_pattern = 'ct_train_{}_image.nii.gz'
    label_pattern = 'ct_train_{}_label.nii.gz'

    # 定义标签映射
    labels = {
        "background": 0,
        "LVM": 1,
        "LAC": 2,
        "LVC": 3,
        "RAC": 4,
        "RVC": 5,
        "AAO": 6,
        "PA": 7
    }

    # 复制文件并生成dataset.json
    num_training_cases = 0
    for i in range(1001, 1021):
        image_filename = image_pattern.format(i)
        label_filename = label_pattern.format(i)

        # 复制图像文件
        shutil.copy(join(source_dir, image_filename), join(imagesTr, f'{i:04d}_0000.nii.gz'))

        # 复制标签文件
        shutil.copy(join(source_dir, label_filename), join(labelsTr, f'{i:04d}.nii.gz'))

        num_training_cases += 1

    # 生成dataset.json
    generate_dataset_json(out_dir, {0: 'CT'}, labels, num_training_cases, '.nii.gz', dataset_name, 'SimpleITKIO')


if __name__ == '__main__':
    source_dir = 'C:/Users/Administrator/Desktop/毕设/nnunet/数据/ct_train'  # 原始数据路径
    base_dir = Path('C:/DeepLearning/nnUNet/nnUNetFrame/DATASET/nnUNet_raw')  # 输出路径
    dataset_name = 'Dataset001_MMWHSCTTrain'  # 数据集名称

    convert_dataset(source_dir, base_dir, dataset_name)