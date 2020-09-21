from datasets.videodataset import VideoDataset
from datasets.videodataset_multiclips import (VideoDatasetMultiClips,collate_fn)
from datasets.loader import VideoLoader
from torchvision import get_image_backend

def image_name_formatter(x):
	return f'image_{x:05d}.jpg'

def get_training_data(video_path,
                      annotation_path,
                      dataset_name,
                      input_type,
                      file_type,
                      spatial_transform=None,
                      temporal_transform=None,
                      target_transform=None):
	assert dataset_name in['kinetics','ucf101','hmdb51']
	assert input_type in ['rgb','flow']
	assert file_type in ['jpg','hdf5']


	# 只考虑输入是 jpg的情况

	if get_image_backend() == 'accimage':
		from datasets.loader import ImageLoaderAccImage
		loader = VideoLoader(image_name_formatter,ImageLoaderAccImage())
	else:
		loader = VideoLoader(image_name_formatter)

	video_path_formatter = (lambda root_path,label,video_id:root_path /'train_jpg'/label / video_id)

	# 只考虑数据集为 kinetics或ucf hmbd

	training_data = VideoDataset(video_path,
	                             annotation_path,
	                             'train',
	                             spatial_transform=spatial_transform,
	                             temporal_transform=temporal_transform,
	                             target_transform=target_transform,
	                             video_loader=loader,
	                             video_path_formatter=video_path_formatter
	                             )

	return training_data

def get_validation_data(video_path,
                        annotation_path,
						dataset_name,
                        input_type,
                        file_type,
                        spatial_transform=None,
                        temporal_transform=None,
                        target_transform=None):
	assert dataset_name in ['kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit']
	assert input_type in ['rgb', 'flow']
	assert file_type in ['jpg', 'hdf5']

	loader = VideoLoader(image_name_formatter)
	video_path_formatter = (
		lambda root_path, label, video_id: root_path /'val_jpg'/label / video_id)


	validation_data = VideoDatasetMultiClips(
		video_path,
		annotation_path,
		'val',
		spatial_transform=spatial_transform,
		temporal_transform=temporal_transform,
		target_transform=target_transform,
		video_loader=loader,
		video_path_formatter=video_path_formatter)

	return validation_data, collate_fn


def get_inference_data(video_path,
                        annotation_path,
						dataset_name,
                        input_type,
                        file_type,
                        inference_subset,
                        spatial_transform=None,
                        temporal_transform=None,
                        target_transform=None):

	assert inference_subset in ['train', 'val', 'test']
	assert dataset_name in [
		'kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit'
	]
	assert input_type in ['rgb', 'flow']
	assert file_type in ['jpg', 'hdf5']

	loader = VideoLoader(image_name_formatter)
	video_path_formatter = (
		lambda root_path, label, video_id: root_path / label / video_id)

	if inference_subset == 'train':
		subset = 'train'
	elif inference_subset == 'val':
		subset = 'val'
	elif inference_subset == 'test':
		subset = 'testing'

	inference_data = VideoDatasetMultiClips(
		video_path,
		annotation_path,
		subset,
		spatial_transform=spatial_transform,
		temporal_transform=temporal_transform,
		target_transform=target_transform,
		video_loader=loader,
		video_path_formatter=video_path_formatter,
		target_type=['video_id', 'segment'])

	return inference_data, collate_fn