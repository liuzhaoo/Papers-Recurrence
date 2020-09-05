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

	def __init__(self, size,
	             crop_position=None,
	             crop_positions=['c', 'tl', 'tr', 'bl', 'br']):
		self.size = size
		self.crop_position = crop_position
		self.crop_positions = crop_positions

		if crop_position is None:  # 裁剪的位置未指定则随机选
			self.randomize = True
		else:
			self.randomize = False

		self.randomize_parameters()

	def __call__(self, img):
		image_width = img.size[0]
		image_height = img.size[1]

		h, w = (self.size, self.size)  # 指定要得到的形状

		if self.crop_position == 'c':
			i = int(round((image_height - h) / 2.))  # 得到中心裁剪的图像顶部到原图像顶部的距离
			j = int(round((image_width - w) / 2.))  # 得到中心裁剪的图像左侧到原图像左侧的距离
		# 其实i,j是裁剪后的图像的左上角与原图像左上角的相对坐标

		elif self.crop_position == 'tl':
			i = 0
			j = 0
		elif self.crop_position == 'tr':
			i = 0
			j = image_width - w
		elif self.crop_position == 'bl':
			i = image_height - h
			j = 0
		elif self.crop_position == 'br':
			i = image_height - h
			j = image_width - w

		img = F.crop(img, i, j, h, w)

		return img

	def randomize_parameters(self):
		if self.randomize:
			self.crop_position = self.crop_positions[random.randint(
				0, len(self.crop_positions) - 1
			)]  # 在0-4之间随机选择

	def __repr__(self):
		return self.__class__.__name__ + '(size={0},crop_position={1},randomize={2}'.format(
			self.size, self.crop_position, self.randomize
		)  # 用来显示信息


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
	def __init__(self, p=0.5):
		super().__init__(p)
		self.randomize_parameters()

	def __call__(self, img):
		if self.random_p < self.p:
			return F.hflip(img)

		return img

	def randomize_parameters(self):
		self.random_p = random.random()


class MultiScaleCornerCrop(object):
	def __init__(self,
	             size,
	             scales,
	             crop_positions=['c', 'tl', 'tr', 'bl', 'br'],
	             interpolation=Image.BILINEAR):
		self.size = size
		self.scales = scales
		self.interpolation = interpolation
		self.crop_positions = crop_positions

		self.randomize_parameters()

	def __call__(self, img):
		short_side = min(img.size[0], img.size[1])
		crop_size = int(short_side * self.scale)

		self.corner_crop.size = crop_size  # 将缩放后的size作为 函数corner_crop的输入的size

		img = self.corner_crop(img)  # 对图像进行角落裁剪，目标size是根据self.scale缩放后的size

		return img.resize((self.size, self.size), self.interpolation)  # 对缩放后的图像还要resize

	def randomize_parameters(self):
		self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
		crop_position = self.crop_positions[random.randint(0, len(self.crop_positions) - 1)]

		self.corner_crop = CornerCrop(None, crop_position)

	def __repr__(self):
		return self.__class__.__name__ + '(size={0},scales={1},interpolation={2})'.format(self.size, self.scales,
		                                                                                  self.interpolation)


class RandomResizeCrop(transforms.RandomResizedCrop):
	def __init__(self,
	             size,
	             scale=(0.08, 1.0),
	             ratio=(3. / 4, 4. / 3),
	             interpolation=Image.BILINEAR):
		super().__init__(size, scale, ratio, interpolation)

		self.randomize_parameters()

	def __call__(self, img):
		if self.randomize:
			self.random_crop = self.get_params(img, self.scale, self.ratio)
			self.randomize = False

		i, j, h, w = self.random_crop

		return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

	def randomize_parameters(self):
		self.randomize = True


class ColorJitter(transforms.ColorJitter):
	def __init__(self,
	             brightness=0,
	             contrast=0,
	             saturation=0,
	             hue=0):
		super().__init__(brightness,contrast,saturation,hue)

		self.randomize_parameters()

	def __call__(self, img):
		if self.randomize:
			self.transform = self.get_params(self.brightness,self.contrast,self.saturation,self.hue)
			self.randomize=False

		return self.transform(img)

	def randomize_parameters(self):
		self.randomize = True



class PickFirstChannels(object):

	def __init__(self,n):
		self.n=n

	def __call__(self,tensor):
		return tensor[:self.n,:,:]

	def randomize_parameters(self):
		pass