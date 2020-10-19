import pandas as pd
import pathlib
from pathlib import Path
from sklearn.utils import shuffle
root_train = Path('../subset/jpg/train')
root_val = Path('../subset/jpg/val')


labels = [x.stem for x in root_train.iterdir()]
# print(labels)
num_t = 0
num_v = 0

c1_t = []
c2_t = []
c3_t = []

c1_v = []
c2_v = []
c3_v = []

for label in labels:
	current_path_t = root_train/label
	current_path_v = root_val/label

	# num_t = len([x for x in current_path_t.iterdir()])
	num_t =100

	# num_v = len([x for x in current_path_v.iterdir()])
	num_v = 6


	col1_t = [label]*num_t
	col2_t = [x.stem for x in current_path_t.iterdir()]

	col2_t = col2_t[:100]
	clo3_t = ['train']*num_t
	c1_t += col1_t
	c2_t += col2_t
	c3_t += clo3_t

	col1_v = [label]*num_v
	col2_v = [x.stem for x in current_path_v.iterdir()]

	col2_v = col2_v[:6]
	clo3_v = ['val']*num_v
	c1_v += col1_v
	c2_v += col2_v
	c3_v += clo3_v


csv_file_train = pd.DataFrame({'label':c1_t,'youtubeid':c2_t,'spilt':c3_t})
csv_file_train =shuffle(csv_file_train)
csv_file_train.to_csv('train_subset_100-6.csv',index=False,sep=',')

csv_file_val = pd.DataFrame({'label':c1_v,'youtubeid':c2_v,'spilt':c3_v})
csv_file_val = shuffle(csv_file_val)
csv_file_val.to_csv('val_subset1_100-6.csv',index=False,sep=',')


# a = [1,2,3]
# b = [4,5,6]
#
# #字典中的key值即为csv中列名
# dataframe = pd.DataFrame({'a_name':a,'b_name':b})
#
# #将DataFrame存储为csv,index表示是否显示行名，default=True
# dataframe.to_csv("test.csv",index=False,sep=',')

