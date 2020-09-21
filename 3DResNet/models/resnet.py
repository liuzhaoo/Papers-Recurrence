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
	expansion = 4  # 进行中间层的一些特征图的扩张

	def __init__(self, in_planes, planes, stride=1, downsample=None):
		super().__init__()

		self.conv1 = conv1x1x1(in_planes, planes)
		self.bn1 = nn.BatchNorm3d(planes)
		self.conv2 = conv3x3x3(planes, planes, stride)
		self.bn2 = nn.BatchNorm3d(planes)
		self.conv3 = conv1x1x1(planes, planes * self.expansion)
		self.bn3 = nn.BatchNorm3d(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity

		out = self.relu(out)

		return out


class ResNet(nn.Module):

	def __init__(self,
	             block,
	             layers,
	             block_inplanes,
	             no_max_pool=False,
	             shortcut_type='B',
	             widen_factor=1.0,
	             n_classes=400):
		super().__init__()

		block_inplanes = [int(x * widen_factor) for x in block_inplanes]  # 每个block的输入

		self.in_planes = block_inplanes[0]  # 第一个layer的输入
		self.no_max_pool = no_max_pool

		self.conv1 = nn.Conv3d(3, self.in_planes,
		                       kernel_size=(7, 7, 7),
		                       stride=(1, 2, 2),
		                       padding=(7//2, 3, 3),
		                       bias=False)  # 在第一个卷积层对空间维度进行一次下采样，时间维度不变

		self.bn1 = nn.BatchNorm3d(self.in_planes)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)  # 下采样

		self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
		                               shortcut_type)  # block_inplances[0]=64,layers[0] 是此layer的block数
		self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], shortcut_type, stride=2)
		self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], shortcut_type, stride=2)
		self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], shortcut_type, stride=2)

		self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  # 通过平均池化将时间和空间的都转换为1

		self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				nn.init.kaiming_normal_(m.weight,
				                        mode='fan_out',
				                        nonlinearity='relu')

			elif isinstance(m, nn.BatchNorm3d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _downsample_basic_block(self, x, planes, stride):
		out = F.avg_pool3d(x, kernel_size=1, stride=stride)
		zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4))
		if isinstance(out.data, torch.cuda.FloatTensor):
			zero_pads = zero_pads.cuda()

		out = torch.cat([out.data, zero_pads], dim=1)  # 将多出来的维度用0填充

		return out

	def _make_layer(self, block, planes, blocks, shorcut_type, stride=1):

		downsample = None
		if stride != 1 or self.in_planes != planes * block.expansion:  # 对identity下采样的条件
			if shorcut_type == 'A':
				downsample = partial(self._downsample_basic_block, planes=planes * block.expansion, stride=stride)

			else:
				downsample = nn.Sequential(
					conv1x1x1(self.in_planes, planes * block.expansion, stride),
					nn.BatchNorm3d(planes * block.expansion)
				)

		layers = []
		layers.append(
			block(in_planes=self.in_planes,
			      planes=planes,
			      stride=stride,
			      downsample=downsample))  # 每一层的第一个block
		self.in_planes = planes * block.expansion  # 每一层的第一个block之后，将self.in_planes 变为输出的4倍

		for i in range(1, blocks):
			layers.append(block(self.in_planes, planes))  # 此时self.in_planes 是planes的4倍，block产生的几个卷积层的总体输入和输出相同

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)  # 空间下采样，时间不变
		x = self.bn1(x)
		x = self.relu(x)
		if not self.no_max_pool:
			x = self.maxpool(x)  # 若参数为None,进行最大池化

		x = self.layer1(x)
		x = self.layer2(x)  # downsample
		x = self.layer3(x)  # downsample
		x = self.layer4(x)  # downsample

		x = self.avgpool(x)  #

		x = x.view(x.size(0), -1)  # 转换维度,[batch_size,block_inplanes[3] * block.expansion]

		x = self.fc(x)  # size[batch,n_classes]

		return x


def generate_model(model_depth, **kwargs):
	assert model_depth in [10, 18, 34, 50, 101, 152, 200]

	if model_depth == 10:
		model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
	elif model_depth == 18:
		model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
	elif model_depth == 34:
		model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
	elif model_depth == 50:
		model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
	elif model_depth == 101:
		model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
	elif model_depth == 152:
		model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
	elif model_depth == 200:
		model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

	return model
