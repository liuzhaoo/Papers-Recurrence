import torch
import torch.nn as nn


class Conv3d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
		super(Conv3d, self).__init__()
		self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
		self.bn = nn.BatchNorm3d(out_channels)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		x = self.conv3d(x)
		x = self.bn(x)
		x = self.relu(x)

		return x


class Stem(nn.Module):
	def __init__(self, in_channels):
		super(Stem, self).__init__()
		self.conv1 = Conv3d(in_channels, 16, 3, stride=(1, 2, 2), padding=(1, 0, 0))  # 空间下采样 16x23x149x149
		self.conv2 = Conv3d(16, 32, 3, stride=(1, 1, 1), padding=(1, 0, 0))  # 32x23x147x147
		self.conv3 = Conv3d(32, 64, 3, stride=(1, 1, 1), padding=(1, 1, 1))  # 64x23x147x147
		self.maxpool1 = nn.MaxPool3d(3, stride=2, padding=0)  # 64x11x73x73

		self.conv4 = Conv3d(64, 128, 3, stride=1, padding=(1, 0, 0))  # 128x11x71x71
		self.conv5 = Conv3d(128, 128, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0))  # 128x11x71x71
		self.conv6 = Conv3d(128, 128, kernel_size=(1, 7, 1), stride=(1, 1, 1), padding=(0, 3, 0))  # 128x11x71x71
		self.conv7 = Conv3d(128, 128, kernel_size=(1, 1, 7), stride=(1, 1, 1), padding=(0, 0, 3))  # 128x11x71x71
		self.maxpool2 = nn.MaxPool3d(3, stride=2, padding=0)  # 128x5x35x35
		self.conv8 = Conv3d(128, 256, 1, stride=1, padding=0)  # 256x5x35x35

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.maxpool1(x)
		x = self.conv4(x)
		x = self.conv5(x)
		x = self.conv6(x)
		x = self.conv7(x)
		x = self.maxpool2(x)
		x = self.conv8(x)

		x_size = x.size()
		return x


class Block_A_1(nn.Module):
	def __init__(self, in_channels):
		super(Block_A_1, self).__init__()
		self.branch0 = nn.Sequential(
			Conv3d(in_channels, 32, 1, stride=1, padding=0),  # 32x5x35x35
			Conv3d(32, 32, 3, stride=2, padding=0),  # 32x2x17x17
			Conv3d(32, 32, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1)),  # 32x2x17x17
			Conv3d(32, 32, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0)),  # 32x2x17x17
			Conv3d(32, 32, 1, stride=1, padding=0)  # 32x2x17x17
		)  # 32x2x17x17

		self.branch1 = nn.Sequential(
			Conv3d(in_channels, 32, 1, stride=1, padding=0),  # 32x5x35x35
			Conv3d(32, 64, 3, stride=2, padding=0),  # 64x2x17x17
		)
		self.branch1_res = nn.Sequential(
			Conv3d(64, 64, 1, stride=1, padding=0),  # 64x2x17x17
			Conv3d(64, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),  # 128x2x17x17
			nn.Conv3d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),  # 64x2x17x17
			nn.BatchNorm3d(64)
		)  # 64x2x17x17

		self.relu = nn.ReLU(inplace=True)
		self.conv = nn.Conv3d(96, 256, 1, stride=1, padding=0, bias=True)

	def forward(self, x):
		x0 = self.branch0(x)  # 32x2x17x17
		x1_1 = self.branch1(x)
		x1_res = self.branch1_res(x1_1)
		x1 = x1_1 + x1_res
		x1 = self.relu(x1)  # 64x2x17x17

		x = torch.cat((x0, x1), dim=1)
		x = self.conv(x)  # 256x2x17x17
		return x


class Block_A(nn.Module):
	def __init__(self, in_channels):
		super(Block_A, self).__init__()
		self.branch0 = nn.Sequential(
			Conv3d(in_channels, 32, 1, stride=1, padding=0),  # 32x2x17x17
			Conv3d(32, 32, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),  # 32x2x17x17
			Conv3d(32, 32, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1)),  # 32x2x17x17
			Conv3d(32, 32, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0)),  # 32x2x17x17
			Conv3d(32, 32, 1, stride=1, padding=0)  # 32x2x17x17
		)  # 32x2x17x17

		self.branch1 = nn.Sequential(
			Conv3d(in_channels, 32, 1, stride=1, padding=0),  # 32x2x17x17
			Conv3d(32, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),  # 64x2x17x17
		)
		self.branch1_res = nn.Sequential(
			Conv3d(64, 64, 1, stride=1, padding=0),  # 64x2x17x17
			Conv3d(64, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),  # 128x2x17x17
			nn.Conv3d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),  # 64x2x17x17
			nn.BatchNorm3d(64)
		)  # 64x2x17x17

		self.relu = nn.ReLU(inplace=True)
		self.conv = nn.Conv3d(96, 256, 1, stride=1, padding=0, bias=True)

	def forward(self, x):
		x0 = self.branch0(x)  # 32x2x17x17
		x1_1 = self.branch1(x)
		x1_res = self.branch1_res(x1_1)
		x1 = x1_1 + x1_res
		x1 = self.relu(x1)  # 64x2x17x17

		x = torch.cat((x0, x1), dim=1)
		x = self.conv(x)  # 256x2x17x17
		return x


class downsample(nn.Module):
	def __init__(self, in_channels):
		super(downsample, self).__init__()
		self.branch0 = nn.MaxPool3d(3, stride=2, padding=(1, 0, 0))  # 256x1x8x8
		self.branch1 = nn.Sequential(
			Conv3d(256, 128, 1, stride=1, padding=0),  # 128x2x17x17
			Conv3d(128, 512, 3, stride=2, padding=(1, 0, 0))  # 512x1x8x8
		)

		self.conv = nn.Conv3d(768, 896, 1, stride=1, padding=0, bias=True)

	def forward(self, x):
		x0 = self.branch0(x)  # 256x1x8x8
		x1 = self.branch1(x)  # 512x1x8x8
		x = torch.cat((x0, x1), dim=1)  # 768x1x8x8
		x = self.conv(x)  # 896x1x8x8
		return x


class Bolock_B(nn.Module):
	def __init__(self, in_channels, scale=1.0):
		super(Bolock_B, self).__init__()
		self.scale = scale
		self.branch0 = Conv3d(in_channels, 128, 1, stride=1, padding=0)  # 128x1x8x8
		self.branch1 = nn.Sequential(
			Conv3d(in_channels, 128, 1, stride=1, padding=0),  # 256x1x8x8
			Conv3d(128, 256, (1, 1, 3), stride=1, padding=(0, 0, 1)),  # 256x1x8x8
			Conv3d(256, 256, (1, 3, 1), stride=1, padding=(0, 1, 0))  # 256x1x8x8
		)
		self.branch2 = nn.Sequential(
			Conv3d(in_channels, 128, 1, stride=1, padding=0),  # 256x1x8x8
			Conv3d(128, 256, (1, 1, 5), stride=1, padding=(0, 0, 2)),  # 256x1x8x8
			Conv3d(256, 256, (1, 5, 1), stride=1, padding=(0, 2, 0))  # 256x1x8x8
		)
		self.conv = nn.Conv3d(640, 896, 1, stride=1, padding=0, bias=True)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		x0 = self.branch0(x)  # 128x1x8x8
		x1 = self.branch1(x)  # 256x1x8x8
		x2 = self.branch2(x)  # 256x1x8x8

		x_res = torch.cat((x0, x1, x2), dim=1)  # 640x1x8x8
		x_res = self.conv(x_res)  # 896x1x8x8

		return self.relu(x + x_res * self.scale)


class LAU_Net(nn.Module):
	def __init__(self, in_channels=3, classes=400):
		super(LAU_Net, self).__init__()
		blocks = []
		blocks.append(Stem(in_channels))
		blocks.append(Block_A_1(256))
		for i in range(4):
			blocks.append(Block_A(256))
		blocks.append(downsample(256))
		for i in range(4):
			blocks.append(Bolock_B(896))

		self.features = nn.Sequential(*blocks)

		self.global_average_poolong = nn.AdaptiveAvgPool3d((1, 1, 1))

		self.fc = nn.Linear(896, classes)

	def forward(self, x):
		x = self.features(x)
		x = self.global_average_poolong(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x


def gennerate_model(classes=101,**kwargs):
	model = LAU_Net(in_channels=3, classes=classes, **kwargs)

	return model
