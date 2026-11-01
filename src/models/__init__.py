# -*- coding: utf-8 -*-
"""深度学习模型：基线网络（ReconNet）与自研网络（OursNet）。

接口约定：`samples_internally` 类属性区分两类模型——
    False（ReconNet）: 前向接受测量值 (B,M)，采样在外部用固定 Φ 完成；
    True （OursNet） : 前向接受图像块 (B,1,33,33)，采样是网络的一部分（可学习）。
训练与评估脚本据此分支。
"""
from __future__ import annotations

import torch.nn as nn

from .ours import OursNet
from .reconnet import ReconNet

__all__ = ["ReconNet", "OursNet", "build_model"]


def build_model(name: str, m: int, **kwargs) -> nn.Module:
    """模型工厂。name: reconnet | ours"""
    if name == "reconnet":
        return ReconNet(m, **kwargs)
    if name == "ours":
        return OursNet(m, **kwargs)
    raise ValueError(f"未知模型: {name}")
