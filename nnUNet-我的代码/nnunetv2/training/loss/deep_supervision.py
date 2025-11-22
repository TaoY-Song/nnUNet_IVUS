import torch
from torch import nn
import numpy as np

# class DeepSupervisionWrapper(nn.Module):
#     def __init__(self, loss, weight_factors=None):
#         """
#         Wraps a loss function so that it can be applied to multiple outputs. Forward accepts an arbitrary number of
#         inputs. Each input is expected to be a tuple/list. Each tuple/list must have the same length. The loss is then
#         applied to each entry like this:
#         l = w0 * loss(input0[0], input1[0], ...) +  w1 * loss(input0[1], input1[1], ...) + ...
#         If weights are None, all w will be 1.
#         """
#         super(DeepSupervisionWrapper, self).__init__()
#         assert any([x != 0 for x in weight_factors]), "At least one weight factor should be != 0.0"
#         self.weight_factors = tuple(weight_factors)
#         self.loss = loss

#     def forward(self, *args):
#         assert all([isinstance(i, (tuple, list)) for i in args]), \
#             f"all args must be either tuple or list, got {[type(i) for i in args]}"
#         # we could check for equal lengths here as well, but we really shouldn't overdo it with checks because
#         # this code is executed a lot of times!

#         if self.weight_factors is None:
#             weights = (1, ) * len(args[0])
#         else:
#             weights = self.weight_factors

#         return sum([weights[i] * self.loss(*inputs) for i, inputs in enumerate(zip(*args)) if weights[i] != 0.0])

class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors=None):
        super().__init__()
        # 修改点：显式检查None而不是用or操作符
        if weight_factors is None:
            self.weight_factors = [1.0]
        else:
            # 转换NumPy数组为列表
            if isinstance(weight_factors, np.ndarray):
                self.weight_factors = weight_factors.tolist()
            else:
                self.weight_factors = list(weight_factors)  # 确保是列表类型
        
        # 验证权重因子
        assert all(w >= 0 for w in self.weight_factors), "权重因子必须非负"
        assert any(w > 0 for w in self.weight_factors), "至少需要一个正权重因子"
        
        self.loss = loss
        self._deep_mode = True

    def forward(self, *args):
        # 自动检测输入结构
        if not self._deep_mode or any(not isinstance(x, (list, tuple)) for x in args):
            # 单层模式：直接传递参数
            return self.loss(*args)
        
        # 多层模式：逐层计算损失
        losses = []
        for i, layer_args in enumerate(zip(*args)):
            if i >= len(self.weight_factors) or self.weight_factors[i] == 0:
                continue
            layer_loss = self.loss(*layer_args)
            losses.append(self.weight_factors[i] * layer_loss)
        
        return sum(losses) if losses else torch.tensor(0.0, device=args[0][0].device)

    def enable_deep_supervision(self, enabled: bool):
        """ 动态切换深度监督模式 """
        self._deep_mode = enabled
