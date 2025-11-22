from __future__ import annotations
import multiprocessing
import os
from typing import List
from pathlib import Path
from warnings import warn

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, subfiles
from nnunetv2.configuration import default_num_processes


def _convert_to_npy(npz_file: str, unpack_segmentation: bool = True, overwrite_existing: bool = False,
                    verify_npy: bool = False, fail_ctr: int = 0) -> None:
    data_npy = npz_file[:-3] + "npy"
    seg_npy = npz_file[:-4] + "_seg.npy"
    previous_npy = npz_file[:-4] + "_previous.npy"  # 前一帧数据的 .npy 文件路径
    try:
        npz_content = None  # will only be opened on demand

        if overwrite_existing or not isfile(data_npy):
            try:
                npz_content = np.load(npz_file) if npz_content is None else npz_content
            except Exception as e:
                print(f"无法打开预处理文件 {npz_file}。请重新运行 nnUNetv2_preprocess！")
                raise e
            np.save(data_npy, npz_content['current_data'])  ##############原本是 np.save(data_npy, npz_content['data'])

        if unpack_segmentation and (overwrite_existing or not isfile(seg_npy)):
            try:
                npz_content = np.load(npz_file) if npz_content is None else npz_content
            except Exception as e:
                print(f"无法打开预处理文件 {npz_file}。请重新运行 nnUNetv2_preprocess！")
                raise e
            np.save(npz_file[:-4] + "_seg.npy", npz_content['current_seg'])  ##############原本是 np.save(npz_file[:-4] + "_seg.npy", npz_content['seg'])

        # 保存前一帧数据到 .npy 文件
        if overwrite_existing or not isfile(previous_npy):
            try:
                npz_content = np.load(npz_file) if npz_content is None else npz_content
            except Exception as e:
                print(f"无法打开预处理文件 {npz_file}。请重新运行 nnUNetv2_preprocess！")
                raise e
            np.save(previous_npy, npz_content['previous_data'])  # 保存前一帧数据

        if verify_npy:
            try:
                np.load(data_npy, mmap_mode='r')
                np.load(previous_npy, mmap_mode='r')  # 验证前一帧数据
                if isfile(seg_npy):
                    np.load(seg_npy, mmap_mode='r')
            except ValueError:
                os.remove(data_npy)
                os.remove(seg_npy)
                os.remove(previous_npy)
                print(f"检查 {data_npy}、{seg_npy} 和 {previous_npy} 时出错，正在修复...")
                if fail_ctr < 2:
                    _convert_to_npy(npz_file, unpack_segmentation, overwrite_existing, verify_npy, fail_ctr + 1)
                else:
                    raise RuntimeError("无法修复解压缩。请检查您的系统或重新运行 nnUNetv2_preprocess")


    except KeyboardInterrupt:
        if isfile(data_npy):
            os.remove(data_npy)
        if isfile(seg_npy):
            os.remove(seg_npy)
        if isfile(previous_npy):
            os.remove(previous_npy)
        raise KeyboardInterrupt


def unpack_dataset(folder: str, unpack_segmentation: bool = True, overwrite_existing: bool = False,
                   num_processes: int = default_num_processes,
                   verify_npy: bool = False):
    """
    all npz files in this folder belong to the dataset, unpack them all
    """
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        npz_files = subfiles(folder, True, None, ".npz", True)
        p.starmap(_convert_to_npy, zip(npz_files,
                                       [unpack_segmentation] * len(npz_files),
                                       [overwrite_existing] * len(npz_files),
                                       [verify_npy] * len(npz_files))
                  )


def get_case_identifiers(folder: str) -> List[str]:
    """
    finds all npz files in the given folder and reconstructs the training case names from them
    """
    case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npz") and (i.find("segFromPrevStage") == -1)]
    return case_identifiers


if __name__ == '__main__':
    unpack_dataset('/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/2d')
