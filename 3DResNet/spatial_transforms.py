import random
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
from PIL import Image


class Compose(transforms.Compose):
	def randomize_parameters(self):
		for t in self.transforms:
			t.randomize_parameters()


class ToTensor(transforms.ToTensor):

	def randomize_parameters(self):
		pass


class Normalize(transforms.Normalize):

	def randomize_parameters(self):
		pass


class ScaleValue(object):
	def __init__(self, s):
		self.s = s

	def __call__(self, tensor):
		tensor *= self.s  # 进行尺度缩放，s是缩放参数
		return tensor

	def randomize_parameters(self):
		pass


class Resize(transforms.Resize):

	def randomize_parameters(self):
		pass


class Scale(transforms.Scale):

	def randomize_parameters(self):
		pass


class CenterCrop(transforms.CenterCrop):

	def randomize_parameters(self):
		pass


class CornerCrop(object):


	def __init__(self,size,
	             crop_position=None,
	             crop_positions=['c','tl','tr','bl','br']):
		self.size = size
		self.crop_position = crop_position
		self.crop_positions = crop_positions

