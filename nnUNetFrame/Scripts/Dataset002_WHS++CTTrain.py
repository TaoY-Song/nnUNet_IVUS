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

    # 定义标签映射
    labels = {
        "background": 0,
        "Myo": 1,
        "LA": 2,
        "LV": 3,
        "RA": 4,
        "RV": 5,
        "AO": 6,
        "PA": 7
    }

    # 复制文件并生成dataset.json
    num_training_cases = 0
    for center_folder in ["CenterA", "CenterB"]:
        center_path = os.path.join(source_dir, center_folder)
        for case_folder in os.listdir(center_path):
            case_num = int(case_folder[4:])

            # 定义图像和标签文件名
            image_filename = f"{case_folder}_image.nii.gz"
            label_filename = f"{case_folder}_label.nii.gz"

            # 复制图像文件
            shutil.copy(join(center_path, case_folder, image_filename), join(imagesTr, f'{case_num:04d}_0000.nii.gz'))

            # 复制标签文件
            shutil.copy(join(center_path, case_folder, label_filename), join(labelsTr, f'{case_num:04d}.nii.gz'))

            num_training_cases += 1

    # 生成dataset.json
    generate_dataset_json(out_dir, {0: 'CT'}, labels, num_training_cases, '.nii.gz', dataset_name, 'SimpleITKIO')


if __name__ == '__main__':
    source_dir = 'C:/Users/Administrator/Desktop/ct_train'  # 原始数据路径
    base_dir = Path('C:/DeepLearning/nnUNet/nnUNetFrame/DATASET/nnUNet_raw')  # 输出路径
    dataset_name = 'Dataset002_WHS++CTTrain'  # 数据集名称

    convert_dataset(source_dir, base_dir, dataset_name)