from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
	return [64, 128, 256, 512]


def conv3x3x3(in_planes,out_planes,stride=1):
	return nn.Conv3d(in_planes,out_planes,
	                 kernel_size=3,
	                 stride=stride,
	                 padding=1,
	                 bias=False)      # 在3个维度的卷积核都是3，此层卷积不改变特征图大小

def conv1x1x1(in_planes,out_planes,stride=1):
	return nn.Conv3d(in_planes,
	                 out_planes,
	                 kernel_size=3,
	                 stride=stride,
	                 bias=False)   # padding 默认为0
