import argparse
import json
from pathlib import Path
import pandas as pd
import re


def get_n_frames(video_path):
	return len([
		x for x in video_path.iterdir()  # 在video子目录中循环
		if 'image' in x.name and x.name[0] != '.'
	])


def convert_csv_to_dict(csv_path,subset):

	data = pd.read_csv(csv_path)
	keys = []
	key_labels = []
	for i in range(data.shape[0]):  # 0即代表行
		row = data.iloc[i, :]  # 取一行的所有内容
		# 得到
		basename = '%s' % row["youtubeid"]  #
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


def load_labels(csv_path):
	data = pd.read_csv(csv_path)
	return data['label'].unique().tolist()  # 将所有类放到列表里


def convert_kinetics_csv_to_json(train_csv_path, val_csv_path,
                                 video_dir_path_t, video_dir_path_v,video_type, dst_json_path):
	labels = load_labels(train_csv_path)
	dst_data = {}
	dst_data['labels'] = labels
	dst_data['database'] = {}
	train_database = convert_csv_to_dict(train_csv_path,'train')
	val_database = convert_csv_to_dict(val_csv_path,'val')
	# if test_csv_path.exists():
	# 	test_database = convert_csv_to_dict(test_csv_path)


	dst_data['database'].update(train_database)
	dst_data['database'].update(val_database)
	# if test_csv_path.exists():
	# 	dst_data['database'].update(test_database)

	for k, v in dst_data['database'].items():
		if 'label' in v['annotations']:
			label = v['annotations']['label']  # 取出每个视频的label
		else:
			label = 'test'

		label = re.sub(r' ','_',label)         # 将类别中的空格改为 _

		if video_type == 'jpg':
			video_path1 = video_dir_path_t / label / k
			video_path2 = video_dir_path_v / label / k
			if video_path1.exists():
				n_frames = get_n_frames(video_path1)  # 计算帧数量
				v['annotations']['segment'] = (1, n_frames + 1)
			elif video_path2.exists():
				n_frames = get_n_frames(video_path2)  # 计算帧数量
				v['annotations']['segment'] = (1, n_frames + 1)




	with dst_json_path.open('w') as dst_file:
		json.dump(dst_data, dst_file)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--dir_path', default='/home/lzhao/FILE/datasets/kinetics/subset/files', type=Path, help='输入csv路径'

	)
	parser.add_argument(
		'--video_patht', default='/home/lzhao/FILE/datasets/kinetics/subset/jpg/train', type=Path, help='照片数据'    # 每次修改
	)
	parser.add_argument(
		'--video_pathv', default='/home/lzhao/FILE/datasets/kinetics/subset/jpg/val', type=Path, help='照片数据'  # 每次修改
	)
	parser.add_argument('--video_type',
	                    default='jpg',
	                    type=str,
	                    help='jpg or hdf5')
	parser.add_argument('--dst_path',
	                    default='/home/lzhao/FILE/datasets/kinetics/subset/kinetics.json',
	                    type=Path,
	                    help='Path of dst json file.')
	args = parser.parse_args()

	train_csv_path = (args.dir_path / 'train_subset.csv')
	val_csv_path = (args.dir_path / 'val_subset.csv')
	test_csv_path = None

	convert_kinetics_csv_to_json(train_csv_path, val_csv_path,
	                             args.video_patht,args.video_pathv, args.video_type, args.dst_path)
