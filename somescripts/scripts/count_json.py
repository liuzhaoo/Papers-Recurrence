import json

path = '/home/lzhao/FILE/datasets/UCF101/ucf101_01.json'
with open(path,'r') as jf:
	load_dict = json.load(jf)
train_num = 0
val_num = 0
for key,value in load_dict['database'].items():
	if value['subset'] == 'training':
		train_num+=1
	elif value['subset'] == 'validation':
		val_num+=1

print('train num = {}'.format(train_num))
print('val num = {}'.format(val_num))


# print(len(load_dict['database']))