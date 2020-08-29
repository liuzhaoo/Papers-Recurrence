import argparse
import json
from pathlib import Path

import pandas as pd
from .utils import get_n_frames, get_n_frames_hdf5


def convert_csv_to_dict(csv_path):
	subset = csv_path.stem
	data = pd.read_csv(csv_path)
	keys = []
	key_labels = []
	for i in range(data.shape[0]):  # 0即代表行
		row = data.iloc[i, :]  # 取一行的所有内容
		# 得到
		basename = '%s_%s_%s_' % (row["youtube_id"], '%06d' % row['time_start'], '%06d' % row['time_end'])  #
		keys.append(basename)

		if subset != 'testing':
			key_labels.append(row['label'])

	database = {}
	for i in range(len(keys)):  # keys 的长度就是该数据集中所有视频的数量
		key = keys[i]
		database[key] = {}  # 以key为键，新建一个字典为值，构成一个键值对
		database[key]['subset'] = subset  # 在新建的字典里建立‘subset’ 与 子集名称的键值对
		if subset != 'testing':
			label = key_labels[i]  # 这里的label是每一类的类别
			database[key]['annotations'] = {'label': label}  # 在新建的字典里再新建一个键值对，值是字典

		else:
			database[key]['annotations'] = {}

	return database


def load_labels(train_csv_path):
	data = pd.read_csv(train_csv_path)
	return data['label'].unique.tolist()  # 将所有类放到列表里


def convert_kinetics_csv_to_json(train_csv_path, val_csv_path, test_csv_path,
                                 video_dir_path, video_type, dst_json_path):
	labels = load_labels(train_csv_path)
	train_database = convert_csv_to_dict(train_csv_path)
	val_database = convert_csv_to_dict(val_csv_path)
	if test_csv_path.exists():
		test_database = convert_csv_to_dict(test_csv_path)

	dst_data = {}
	dst_data['labels'] = labels
	dst_data['database'] = {}
	dst_data['database'].update(train_database)
	dst_data['database'].update(val_csv_path)

	for k, v in dst_data['database'].items():
		if 'label' in v['annotations']:
			label = v['annotations']['label']  # 取出每个视频的label
		else:
			label = 'test'

		if video_type == 'jpg':
			video_path = video_dir_path / label / k
			if video_path.exists():
				n_frames = get_n_frames(video_path)     #计算帧数量
				v['annotations']['segment'] = (1,n_frames+1)
			else:

