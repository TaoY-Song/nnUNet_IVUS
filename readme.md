**基于 nnU-Net v2 适配血管内超声（IVUS）图像分割的改动版**

这是一个对官方 [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) 进行的针对性适配，主要用于 IVUS（血管内超声）的自动分割。

### 主要改动与特性
- 适配 IVUS 图像格式，将一个nifti文件拆分为多帧分别当作2d图像输入
- 引入 空间注意力 SA 模块与时间注意力 TA 模块，提高帧间关联性 
- 更新 `.gitignore`，彻底忽略大体积预处理与模型文件