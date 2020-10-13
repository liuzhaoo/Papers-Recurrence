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
		for i in frame_indices:                                           # 这里的frame_indices是一个视频所有帧的名称里的数字
			image_path = video_path/self.image_name_formatter(i)		 # 一个视频所有帧的路径 video_path是以视频为名的文件夹
			if image_path.exists():
				video.append(self.image_loader(image_path))

		return video                                                     #  因此返回的video是某个视频对应的所有帧