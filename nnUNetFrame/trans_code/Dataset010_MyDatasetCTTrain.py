import os
import shutil
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import save_json, join


def generate_dataset_json(output_folder, channel_names, labels,
                          num_training_cases, file_ending, dataset_name,
                          overwrite_image_reader_writer):
    dataset_json = {
        'channel_names': channel_names,
        'labels': labels,
        'numTraining': num_training_cases,
        'file_ending': file_ending,
        'name': dataset_name,
        'overwrite_image_reader_writer': overwrite_image_reader_writer
    }
    save_json(dataset_json,
              join(output_folder, 'dataset.json'),
              sort_keys=False)


def process_images(source_dir, target_dir):
    for folder_name in os.listdir(source_dir):
        if folder_name.startswith('PA'):
            pa_folder_path = os.path.join(source_dir, folder_name)
            if os.path.isdir(pa_folder_path):
                se_folders = [
                    f for f in os.listdir(pa_folder_path) if f.startswith('SE')
                    and os.path.isdir(os.path.join(pa_folder_path, f))
                ]
                if se_folders:
                    for se_folder in se_folders:
                        se_folder_path = os.path.join(pa_folder_path,
                                                      se_folder)
                        for filename in os.listdir(se_folder_path):
                            if filename.endswith('.nii.gz'):
                                parts = filename.split(' - AcquisitionTime ')
                                if len(parts) >= 2:
                                    file_part = parts[1]
                                else:
                                    file_part = filename
                                new_name = f"{folder_name}_{se_folder}_{file_part}"
                                shutil.copy(
                                    os.path.join(se_folder_path, filename),
                                    os.path.join(target_dir, new_name))
                else:
                    for filename in os.listdir(pa_folder_path):
                        if filename.endswith('.nii.gz'):
                            parts = filename.split(' - AcquisitionTime ')
                            if len(parts) >= 2:
                                file_part = parts[1]
                            else:
                                file_part = filename
                            new_name = f"{folder_name}_{file_part}"
                            shutil.copy(os.path.join(pa_folder_path, filename),
                                        os.path.join(target_dir, new_name))


def process_labels(source_dir, target_dir):
    for folder_name in os.listdir(source_dir):
        if folder_name.startswith('PA') and folder_name.endswith(
                '_labels_postprocessing'):
            pa_folder_path = os.path.join(source_dir, folder_name)
            if os.path.isdir(pa_folder_path):
                se_folders = [
                    f for f in os.listdir(pa_folder_path) if f.startswith('SE')
                    and os.path.isdir(os.path.join(pa_folder_path, f))
                ]
                if se_folders:
                    for se_folder in se_folders:
                        se_folder_path = os.path.join(pa_folder_path,
                                                      se_folder)
                        for filename in os.listdir(se_folder_path):
                            if filename.endswith('.nii.gz'):
                                parts = filename.split(' - AcquisitionTime ')
                                if len(parts) >= 2:
                                    file_part = parts[1]
                                else:
                                    file_part = filename
                                new_name = f"{folder_name}_{se_folder}_{file_part}"
                                new_name = new_name.replace(
                                    '_labels_postprocessing', '')
                                shutil.copy(
                                    os.path.join(se_folder_path, filename),
                                    os.path.join(target_dir, new_name))
                else:
                    for filename in os.listdir(pa_folder_path):
                        if filename.endswith('.nii.gz'):
                            parts = filename.split(' - AcquisitionTime ')
                            if len(parts) >= 2:
                                file_part = parts[1]
                            else:
                                file_part = filename
                            new_name = f"{folder_name}_{file_part}"
                            new_name = new_name.replace(
                                '_labels_postprocessing', '')
                            shutil.copy(os.path.join(pa_folder_path, filename),
                                        os.path.join(target_dir, new_name))


def main():
    # Define directories
    images_source_dir = r'C:\Users\Administrator\Desktop\data_process\NIfTI'
    labels_source_dir = r'C:\Users\Administrator\Desktop\data_process\NIfTI\WHS++_predict_labels'
    base_output_dir = r'C:\Users\Administrator\Desktop'
    dataset_name = 'Dataset010_MyDatasetCTTrain'
    out_dir = os.path.join(base_output_dir, dataset_name)
    imagesTr_dir = os.path.join(out_dir, 'imagesTr')
    labelsTr_dir = os.path.join(out_dir, 'labelsTr')

    # Create directories
    os.makedirs(imagesTr_dir, exist_ok=True)
    os.makedirs(labelsTr_dir, exist_ok=True)

    # Process images
    process_images(images_source_dir, imagesTr_dir)

    # Process labels
    process_labels(labels_source_dir, labelsTr_dir)

    # 初始化训练集文件数量为0
    num_training_cases = 0
    num_training_cases_images = 0
    num_training_cases_labels = 0
    # 获取imagesTr文件夹下的所有文件和文件夹名称列表
    for filename in os.listdir(imagesTr_dir):
        if filename.endswith('.nii.gz'):
            num_training_cases_images += 1

    for filename in os.listdir(labelsTr_dir):
        if filename.endswith('.nii.gz'):
            num_training_cases_labels += 1

    # 比较二者的值，如果相等，将该值赋给num_training_cases
    if num_training_cases_images == num_training_cases_labels:
        num_training_cases = num_training_cases_images
    else:
        print("图像文件数量和标签文件数量不一致，请检查数据！")
        num_training_cases = None

    # Generate dataset.json
    channel_names = {0: 'CT'}
    labels_dict = {
        "background": 0,
        "Myo": 1,
        "LA": 2,
        "LV": 3,
        "RA": 4,
        "RV": 5,
        "AO": 6,
        "PA": 7
    }
    file_ending = '.nii.gz'
    dataset_name_json = 'MyDatasetCTTrain'
    overwrite_image_reader_writer = 'SimpleITKIO'

    generate_dataset_json(out_dir, channel_names, labels_dict,
                          num_training_cases, file_ending, dataset_name_json,
                          overwrite_image_reader_writer)


if __name__ == '__main__':
    main()
