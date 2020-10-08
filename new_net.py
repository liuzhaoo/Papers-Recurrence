import torch
import torch.nn as nn


def conv1x1(in_channel, out_channel, stride=2):
	return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)


class Conv3x3_bn(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1, padding=1, bias=False):
		super(Conv3x3_bn, self).__init__()

		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)
		self.bn = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.relu(x)

		return x


class Conv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, bias=False):
		super(Conv2d, self).__init__()

		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
		self.bn = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.relu(x)

		return x


class Block_A(nn.Module):
	def __init__(self, in_channels, planes, stride=2, downsample=None):
		super(Block_A, self).__init__()
		self.conv1 = Conv3x3_bn(in_channels, planes)
		self.conv2 = Conv3x3_bn(planes, planes)
		self.conv3 = conv1x1(planes, planes, stride=stride)
		self.bn = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample

	def forward(self, x):
		identity = x
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.bn(x)

		if self.downsample is not None:
			identity = self.downsample(identity)

		out = x + identity

		return self.relu(out)


class Block_B(nn.Module):
	def __init__(self, in_channels,  planes, stride=2, downsample=None):
		super(Block_A, self).__init__()
		self.conv1 = Conv3x3_bn(in_channels, planes)
		self.conv2 = Conv3x3_bn(planes, planes)
		self.conv3 = Conv3x3_bn(planes, planes)
		self.conv4 = conv1x1(planes, planes, stride=stride)
		self.bn = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample

	def forward(self, x):
		identity = x
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.bn(x)

		if self.downsample is not None:
			identity = self.downsample(identity)

		out = x + identity

		return self.relu(out)


class Cat_Block(nn.Module):
	def __init__(self, in_channels):
		pass


class New_net(nn.Module):
	def __init__(self, in_channels, num_classes):
		super(New_net, self).__init__()
		self.in_planes = 32
		self.conv1 = Conv2d(in_channels, 32, 7, stride=2, padding=3)  # 32x150x150
		self.layer1 = self.__make_layer(Block_A, 64, stride=2)  # 64x75x75 ,self.in_planes = 64
		self.layer2 = self.__make_layer(Block_A, 128, stride=2)  # 128x38x38,self.in_planes = 128
		self.layer3 = self.__make_layer(Block_B,256,stride=2)  # 256x19x19,self.in_planes=256
		self.layer4 = self.__make_layer(Block_B,512,stride=2)  # 512x10x10,self.in_planes = 512
		self.layer5 = self.__make_layer(Block_A,512,stride=1)  # 512x10x10,self.in_planes = 1024

	def forward(self,x):
		features = []
		x = self.conv1(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)

		return x



	def __make_layer(self, block, planes, stride=2):
		downsample = nn.Sequential(
			conv1x1(self.in_planes, planes, stride=stride),
			nn.BatchNorm2d(planes)
		)

		layers = []
		layers.append(block(in_channels=self.in_planes, planes=planes, stride=stride, downsample=downsample))
		self.in_planes *= 2
		return nn.Sequential(*layers)
