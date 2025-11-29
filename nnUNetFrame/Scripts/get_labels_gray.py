import SimpleITK as sitk
import numpy as np
import os
from tqdm import tqdm


def sliceMain(labf):
    nii_list = [os.path.join(labf, i) for i in os.listdir(labf)]

    cl = []
    for image_nii in tqdm(nii_list):  # 遍历所有的nii文件

        img = sitk.ReadImage(image_nii)
        img_array = sitk.GetArrayFromImage(img)  # nii-->array

        img_num = np.unique(img_array)

        for i in img_num:
            if i not in cl:
                cl.append(i)
    print(cl)
    print(len(cl))


if __name__ == '__main__':
    labels_folder = 'C:\DeepLearning\\nnUNet\\nnUNetFrame\\DATASET\\nnUNet_raw\\Dataset030_MMWHS\\labelsTr'  # 3d nii 的标签数据
    # 切片函数
    sliceMain(labf=labels_folder)