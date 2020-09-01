import json
from pathlib import Path

import torch
import torch.utils.data as data

from .loader import VideoLoader


def get_class_labels(data):
	class_labels_map = {}
	index = 0
	for class_label in data['labels']:
		class_labels_map[class_label] = index  # 将类别与索引对应
		index += 1

	return class_labels_map  # 返回类别的字典


def get_database(data, subset, root_path, video_path_formatter):
	video_ids = []
	video_paths = []
	annotations = []

	for key, value in data['database'].items():
		this_subset = value['subset']
		if this_subset == subset:
			video_ids.append(key)  # 将输入子集的视频名取出来放到video_ids 中
			annotations.append(value['annotations'])  # 视频的标注信息,是一个列表

			if 'video_path' in value:
				video_paths.append(Path(value['video_path']))  # 事实上，value里没有这一项

			else:
				label = value['annotations']['label']
				video_paths.append(video_path_formatter(root_path, label, key))

	return video_ids, video_paths, annotations


class VideoDataset(data.Dataset):
	def __init__(self,
	             root_path,
	             annotation_path,
	             subset,
	             spatial_transform=None,
	             temporal_transform=None,
	             target_transform=None,
	             video_loader=None,
	             video_path_formatter=(lambda root_path, label, video_id:
	             root_path / label / video_id),
	             image_name_formatter=lambda x: f'image_{x:05d}.jpg',
	             target_type='label'):
		self.data, self.class_names = self.__make_dataset(
			root_path, annotation_path, subset, video_path_formatter)

		self.spatial_transform = spatial_transform
		self.temporal_transform = temporal_transform
		self.target_transform = target_transform

		if video_loader is None:
			self.loader = VideoLoader(image_name_formatter)
		else:
			self.loader = video_loader

		self.target_type = target_type

	def __make_dataset(self, root_path, annotation_path, subset, video_path_formatter):
		with annotation_path.open('r') as f:
			data = json.load(f)
		video_ids, video_paths, annotations = get_database(data, subset, root_path, video_path_formatter)
		class_to_idx = get_class_labels(data)
		idx_to_class = {}
		for name, label in class_to_idx.items():
			idx_to_class[label] = name

		n_videos = len(video_ids)
		dataset = []
		for i in range(n_videos):
			if i % (n_videos // 5) == 0:
				print('dataset loading [{}/{}]'.format(i, len(video_ids)))

			if 'label' in annotations[i]:
				label = annotations[i]['label']
				# 取出当前视频的类别，annotation 是按视频名存放的标注信息，顺序与video_ids，video_paths 相同，因此可以通过索引获取对应的信息
				label_id = class_to_idx[label]  # 根据类别得到索引

			else:
				label = 'test'
				label_id = -1

			video_path = video_paths[i]
			if not video_path.exists():
				continue

			segment = annotations[i]['segment']
			if segment[1] == 1:
				continue

			frame_indices = list(range(segment[0], segment[1]))
			sample = {
				'video': video_path,
				'segment': segment,
				'frame_indices': frame_indices,
				'video_id': video_ids[i],
				'label': label_id
			}
			dataset.append(sample)

		return dataset, idx_to_class

	def __loading(self,path,frame_indices):
		clip = self.loader(path,frame_indices)      #  根据路径和id找到每帧图片
		if self.spatial_transform is not None:
			self.spatial_transform.randomize_parameters()
			clip = [self.spatial_transform(img) for img in clip]

		clip = torch.stack(clip,0).permute(1,0,2,3)

		return clip


	def __getitem__(self, index):
		path = self.data[index]['video']
		if isinstance(self.target_type,list)