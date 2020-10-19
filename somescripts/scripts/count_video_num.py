import pandas as pd
import pathlib
from pathlib import Path
from sklearn.utils import shuffle
root_train = Path('../subset/jpg/train')
root_val = Path('../subset/jpg/val')


labels = [x.stem for x in root_train.iterdir()]
min_train_num = 1000
min_val_num = 1000

for label in labels:
	current_path_t = root_train/label
	current_path_v = root_val/label

	num_t = len([x for x in current_path_t.iterdir()])
	if min_train_num > num_t:
		min_train_num = num_t
		path_t = current_path_t



	num_v = len([x for x in current_path_v.iterdir()])
	if min_val_num > num_v:
		min_val_num = num_v
		path_v = current_path_v
	# num_v = 6

print('最少的训练集视频数为{}类，只有{}个'.format(path_t,min_train_num))
print('最少的验证集视频数为{}类，只有{}个'.format(path_v,min_val_num))