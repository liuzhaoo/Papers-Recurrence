from datasets.loader import VideoLoader
from pathlib import Path
import torch
from datasets.videodataset import VideoDataset
def image_name_formatter(x):
    return f'image_{x:05d}.jpg'
#
#
# loader = VideoLoader(image_name_formatter)
# video_path = Path('/home/lzhao/FILE/datasets/kinetics/val_jpg')
# path = Path('../files/kinetics.json')
# # frame_indices = list(range(1,100))
#
#
# # def loading(path, frame_indices):  #
# # 	clip = loader(path, frame_indices)  # 根据路径和id找到每帧图片
# # 	clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
# # 	return clip
# #
# # clip = loader(video_path,frame_indices)
#
#
# training_data = VideoDataset(video_path,
#                              path,
#                              'val')

x1 = torch.randn((3,14,15))