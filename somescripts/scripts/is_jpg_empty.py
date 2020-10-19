import os
from pathlib import Path


root_p = Path('../subset/jpg')
# target = Path('../../train_jpg')
target = Path('../../ucf101')
labels = [x.stem for x in target.iterdir()]

# train = Path('../subset/jpg/train')
# train = Path('../subset/jpg/train')
# val = Path('../subset/jpg/val')
val = Path('../../ucf101')

train_num = 0
train_list = []
val_num = 0
val_list = []

for label in labels:
	# train_label = train / label  # 确定目标类别路径
	val_label = val / label

	# ids_train = [id.stem for id in train_label.iterdir()]  # 获取当前类别包含的所有id
	ids_val = [id.stem for id in val_label.iterdir()]

	# for id in ids_train:
	#
	# 	train_label_id = train_label/id
	#
	# 	names = [name.stem for name in train_label_id.iterdir()]
	# 	print('check train id: {} / label: {}'.format(id,label))
	# 	for name in names:
	# 		name = name+'.jpg'
	# 		file = train_label_id/name
	#
	# 		size = os.path.getsize(file)
	#
	# 		if size == 0:
	# 			emp = 'train  '+label+' '+id
	# 			train_num += 1
	# 			train_list.append(emp)

	for id in ids_val:

		val_label_id = val_label / id

		names = [name.stem for name in val_label_id.iterdir()]
		print('check val id: {} / label: {}'.format(id, label))
		for name in names:
			name = name + '.jpg'
			file = val_label_id / name

			size = os.path.getsize(file)

			if size == 0:
				emp = 'val  '+label + ' ' + id
				val_num += 1
				val_list.append(emp)



print('train_num:{}'.format(train_num))
print('val_num:{}'.format(val_num))

print(train_list)
print(val_list)