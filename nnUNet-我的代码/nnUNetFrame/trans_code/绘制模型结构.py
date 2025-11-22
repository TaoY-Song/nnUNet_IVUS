# import torch
# from torch import nn
# import netron
# from dynamic_network_architectures.architectures.SA&TA_UNet import SAPlainConvUNet
#
# # 实例化网络
# model = SAPlainConvUNet(
#     input_channels=1,
#     n_stages=4,
#     features_per_stage=[32, 64, 128, 256],
#     conv_op=nn.Conv3d,
#     kernel_sizes=3,
#     strides=2,
#     n_conv_per_stage=2,
#     num_classes=2,
#     n_conv_per_stage_decoder=2,
#     conv_bias=False,
#     norm_op=nn.BatchNorm3d,
#     norm_op_kwargs={},
#     dropout_op=None,
#     dropout_op_kwargs={},
#     nonlin=nn.ReLU,
#     nonlin_kwargs={},
#     deep_supervision=False,
#     nonlin_first=False
# )
#
# # 切换到评估模式
# model.eval()
#
# # 生成输入数据
# x_current = torch.randn(2, 1, 16, 16, 16)  # 当前帧
# x_previous = torch.randn(2, 1, 16, 16, 16)  # 前一帧
#
# # 导出为 ONNX 格式
# torch.onnx.export(
#     model,
#     (x_current, x_previous),
#     "model.onnx",
#     input_names=["x_current", "x_previous"],
#     output_names=["output"],
#     opset_version=11
# )
#
# # 使用 Netron 保存为图片
# netron.start("model.onnx")  # 直接截图吧
