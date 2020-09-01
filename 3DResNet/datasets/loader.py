import accimage
from PIL import Image

class ImageLoaderPIL(object):
	#  从路径中打开图片
	def __call__(self,path):
		with path.open('rb') as f:
			with Image.open(f) as img:
				return img.convert('RGB')

class ImageLoaderAccImage(object):

	def __call__(self,path):
		return accimage.Image(str(path))


class VideoLoader(object):
	def __init__(self,image_name_formatter,image_loader=None):
		self.image_name_formatter = image_name_formatter
		if image_loader is None:
			self.image_loader = ImageLoaderPIL
		else:
			self.image_loader = image_loader

	def __call__(self, video_path,frame_indices):
		video = []
		for i in frame_indices:
			image_path = video_path/self.image_name_formatter(i)
			if image_path.exists():
				video.append(self.image_loader(image_path))

		return video