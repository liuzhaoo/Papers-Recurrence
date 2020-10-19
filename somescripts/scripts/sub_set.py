from pathlib import Path
import shutil
import tqdm
root_p = Path('../subset/jpg')
target = Path('../train_jpg')
labels = [x.stem for x in target.iterdir()]


# 生成类别文件夹
# for label in labels:

# 	train_label = root_p/'train'/label
# 	val_label = root_p/'val'/label

	
	#
	# train_label.mkdir()
	# val_label.mkdir()
source_train = Path('../train_jpg')
source_val = Path('../val_jpg')

target_train = Path('../subset/jpg/train')
target_val = Path('../subset/jpg/val')

train_num = 260
val_num =12

for label in labels:
	source_train_label = source_train/label
	source_val_label = source_val/label          #  获取当前类别的路径

	target_train_label =target_train/label     # 确定目标类别路径
	target_val_label =target_val/label

	ids_train = [id.stem for id in source_train_label.iterdir()]   # 获取当前类别包含的所有id
	ids_val = [id.stem for id in source_val_label.iterdir()]

	if len(ids_train) < train_num:
		num_ids_train = ids_train[:len(ids_train)]
	else:
		num_ids_train = ids_train[:train_num]

	num_ids_val = ids_val[:val_num]                            # 截取前260、12 个id

	print('copying label {}'.format(label))
	for id in num_ids_train:

		source_train_label_id = source_train_label/id
		target_train_label_id = target_train_label/id

		if not target_train_label_id.exists():
			target_train_label_id.mkdir()

		print('copying id {} / label {}'.format(id,label))
		for file in source_train_label_id.iterdir():
			shutil.copy(file,target_train_label_id)


	for id in num_ids_val:
		source_val_label_id = source_val_label / id

		target_val_label_id = target_val_label / id
		if not target_val_label_id.exists():
			target_val_label_id.mkdir()

		for file in source_val_label_id.iterdir():
			shutil.copy(str(file),str(target_val_label_id))










