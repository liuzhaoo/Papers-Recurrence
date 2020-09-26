import torchvision

# inceptionv1 = torchvision.models.inception_v3()

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, bias=False):
		super(BasicConv2d, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
		self.bn = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.relu(x)
		return x


class Stem(nn.Module):
	def __init__(self, in_channels):
		super(Stem, self).__init__()

		self.conv2d_1v = BasicConv2d(in_channels, 32, kernel_size=3, stride=2, padding=0)  # Nx32x149x149
		self.conv2d_2v = BasicConv2d(32, 32, 3, stride=1, padding=0)  # Nx32x147x147

		self.conv2d_3 = BasicConv2d(32, 64, 3, stride=1, padding=1)  # Nx64x147x147

		self.mixed_4v_branch_l = nn.MaxPool2d(3, stride=2, padding=0)  # Nx64x73x73
		self.mixed_4v_branch_r = BasicConv2d(64, 96, 3, stride=2, padding=0)  # Nx96x73x73

		# in: 160x73x73
		self.mixed_5_branch_l = nn.Sequential(
			BasicConv2d(160, 64, 1, stride=1, padding=0),  # Nx64x73x73
			BasicConv2d(64, 96, 3, stride=1, padding=0)  # Nx96x71x71
		)
		# in: 160x73x73
		self.mixed_5_branch_r = nn.Sequential(
			BasicConv2d(160, 64, 1, stride=1, padding=0),  # Nx64x73x73
			BasicConv2d(64, 64, (1, 7), stride=1, padding=(0, 3)),  # Nx64x73x73
			BasicConv2d(64, 64, (7, 1), stride=1, padding=(3, 0)),  # Nx64x73x73
			BasicConv2d(64, 96, 3, stride=1, padding=0)  # Nx96x71x71
		)

		# in: 192x71x71
		self.mixed_6v_branch_l = BasicConv2d(192, 192, 3, stride=2, padding=0)  # Nx192x35x35
		self.mixed_6v_branch_r = nn.MaxPool2d(3, stride=2, padding=0)  # Nx192x35x35

	def forward(self, x):
		x = self.conv2d_1v(x)  # Nx32x149x149
		x = self.conv2d_2v(x)  # Nx32x147x147
		x = self.conv2d_3(x)  # Nx64x147x147

		x_4l = self.mixed_4v_branch_l(x)  # Nx64x73x73
		x_4r = self.mixed_4v_branch_r(x)  # Nx96x73x73
		x = torch.cat((x_4l, x_4r), dim=1)  # Nx160x73x73

		x_5l = self.mixed_5_branch_l(x)  # Nx96x71x71
		x_5r = self.mixed_5_branch_r(x)  # Nx96x71x71
		x = torch.cat((x_5l, x_5r), dim=1)  # Nx192x71x71

		x_6l = self.mixed_6v_branch_l(x)  # Nx192x35x35
		x_6r = self.mixed_6v_branch_r(x)  # Nx192x35x35
		x = torch.cat((x_6l, x_6r), dim=1)  # Nx384x35x35

		return x


class Inception_A(nn.Module):
	def __init__(self, in_channels):
		super(Inception_A, self).__init__()

		self.branch_0 = nn.Sequential(
			nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),  # 待查
			BasicConv2d(384, 96, 1, stride=1, padding=0)  # Nx96x35x35
		)

		self.branch_1 = BasicConv2d(in_channels, 96, 1, padding=0)  # Nx96x35x35
		self.branch_2 = nn.Sequential(
			BasicConv2d(in_channels, 64, 1, stride=1, padding=0),
			BasicConv2d(64, 96, 3, stride=1, padding=1)  # Nx96x35x35
		)
		self.branch_3 = nn.Sequential(
			BasicConv2d(in_channels, 64, 1, stride=1, padding=0),
			BasicConv2d(64, 96, 3, stride=1, padding=1),
			BasicConv2d(96, 96, 3, stride=1, padding=1)  # Nx96x35x35
		)

	def forward(self, x):
		x0 = self.branch_0(x)  # Nx96x35x35
		x1 = self.branch_1(x)  # Nx96x35x35
		x2 = self.branch_2(x)  # Nx96x35x35
		x3 = self.branch_3(x)  # Nx96x35x35

		x = torch.cat((x0, x1, x2, x3), dim=1)  # Nx384x35x35
		return x


class Reduction_A(nn.Module):
	def __init__(self, in_channels):
		super(Reduction_A, self).__init__()

		self.branch_0 = nn.MaxPool2d(3, stride=2, padding=0)  # Nx384x17x17
		self.branch_1 = BasicConv2d(in_channels, 384, 3, stride=2, padding=0)  # Nx384x17x17
		self.branch_2 = nn.Sequential(
			BasicConv2d(in_channels, 192, 1, stride=1, padding=1),
			BasicConv2d(192, 224, 3, stride=1, padding=1),
			BasicConv2d(224, 256, 3, stride=2, padding=0)  # Nx256x17x17
		)

	def forward(self, x):
		x0 = self.branch_0(x)
		x1 = self.branch_1(x)
		x2 = self.branch_2(x)

		x = torch.cat((x0, x1, x2), dim=1)  # Nx1024x17x17
		return x


class Inception_B(nn.Module):
	def __init__(self, in_channels):
		super(Inception_B, self).__init__()

		self.branch0 = nn.Sequential(
			# nn.AvgPool2d(3, stride=1, padding=1),
			nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),  # Nx1024x17x17
			BasicConv2d(in_channels, 128, 1, stride=1, padding=0)  # Nx128x17x17
		)
		self.branch1 = BasicConv2d(in_channels, 384, 1, stride=1, padding=0)  # Nx384x17x17
		self.branch2 = nn.Sequential(
			BasicConv2d(in_channels, 192, 1, stride=1, padding=0),  # Nx192x17x17
			BasicConv2d(192, 224, (1, 7), stride=1, padding=(0, 3)),  # Nx224x17x17
			BasicConv2d(224, 256, (7, 1), stride=1, padding=(3, 0)),  # Nx256x17x17
		)

		self.branch3 = nn.Sequential(
			BasicConv2d(in_channels, 192, 1, stride=1, padding=0),  # Nx192x17x17
			BasicConv2d(192, 192, (1, 7), stride=1, padding=(0, 3)),  # Nx192x17x17
			BasicConv2d(192, 224, (7, 1), stride=1, padding=(3, 0)),  # Nx224x17x17
			BasicConv2d(224, 224, (1, 7), stride=1, padding=(0, 3)),  # Nx224x17x17
			BasicConv2d(224, 256, (7, 1), stride=1, padding=(3, 0)),  # Nx256x17x17
		)

	def forward(self, x):
		x0 = self.branch0(x)  # Nx128x17x17
		x1 = self.branch1(x)  # Nx384x17x17
		x2 = self.branch2(x)  # Nx256x17x17
		x3 = self.branch3(x)  # Nx256x17x17

		x = torch.cat((x0, x1, x2, x3), dim=1)  # Nx1024x17x17

		return x


class Reduction_B(nn.Module):
	def __init__(self, in_channels):
		super(Reduction_B, self).__init__()

		self.branch0 = nn.MaxPool2d(3, stride=2, padding=0)  # Nx1024x8x8
		self.branch1 = nn.Sequential(
			BasicConv2d(in_channels, 192, 1, stride=1, padding=0),  # Nx192x17x17
			BasicConv2d(192, 192, 3, stride=2, padding=0),  # Nx192x8x8

		)
		self.branch2 = nn.Sequential(
			BasicConv2d(in_channels, 256, 1, stride=1, padding=0),  # Nx256x17x17
			BasicConv2d(256, 256, (1, 7), stride=1, padding=(0, 3)),  # Nx256x17x17
			BasicConv2d(256, 320, (7, 1), stride=1, padding=(3, 0)),  # Nx320x17x17
			BasicConv2d(320, 320, 3, stride=2, padding=0),  # Nx320x8x8
		)

	def forward(self, x):
		x0 = self.branch0(x)  # Nx1024x8x8
		x1 = self.branch1(x)  # Nx192x8x8
		x2 = self.branch2(x)  # Nx320x8x8

		x = torch.cat((x0, x1, x2), dim=1)  # Nx1536x8x8

		return x


class Inception_C(nn.Module):
	def __init__(self, in_channels):
		super(Inception_C, self).__init__()

		self.branch0 = nn.Sequential(
			nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),  # Nx1536x8x8
			BasicConv2d(1536, 256, 1, stride=1, padding=0)  # Nx256x8x8
		)
		self.branch1_m = BasicConv2d(in_channels, 384, 1, stride=1, padding=0)
		self.branch1_l = BasicConv2d(384, 256, (1, 3), stride=1, padding=(0, 1))
		self.branch1_l = BasicConv2d(384, 256, (3, 1), stride=1, padding=(1, 0))

		self.branch2 = BasicConv2d(in_channels, 256, 1, stride=1, padding=0)
		self.branch3_m = nn.Sequential(
			BasicConv2d(in_channels, 384, 1, stride=1, padding=0),
			BasicConv2d(384, 448, (1, 3), stride=1, padding=(0, 1)),
			BasicConv2d(448, 512, (3, 1), stride=1, padding=(1, 0))
		)
