from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
	return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
	return nn.Conv3d(in_planes, out_planes,
	                 kernel_size=3,
	                 stride=stride,
	                 padding=1,
	                 bias=False)  # 在3个维度的卷积核都是3，此层卷积不改变特征图大小，也不改变时间维度上的长度


def conv1x1x1(in_planes, out_planes, stride=1):
	return nn.Conv3d(in_planes,
	                 out_planes,
	                 kernel_size=1,
	                 stride=stride,
	                 bias=False)  # padding 默认为0，此层也不进行大小的改变


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1, downsample=None):
		super().__init__()

		self.conv1 = conv3x3x3(in_planes, planes, stride)
		self.bn1 = nn.BatchNorm3d(planes)
		self.relu = nn.ReLU(inplace=True)  # inplace 在原地操作
		self.conv2 = conv3x3x3(planes, planes)
		self.bn2 = nn.BatchNorm3d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:  # 若参数中有downsample，则对本层的输入进行下采样
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expeansion = 4  # 进行中间层的一些特征图的扩张

	def __init__(self, in_planes, planes, stride=1, downsample=None):
		super().__init__()

		self.conv1 = conv1x1x1(in_planes, planes)
		self.bn1 = nn.BatchNorm3d(planes)
		self.conv2 = conv3x3x3(planes, planes, stride)
		self.bn2 = nn.BatchNorm3d(planes)
		self.conv3 = conv1x1x1(planes, planes * self.expeansion)
		self.bn3 = nn.BatchNorm3d(planes * self.expeansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride =stride

	def forward(self,x):

		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(x)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(x)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity

		out = self.relu(out)

		return out






