import os 
import pathlib 
from pathlib import Path 

root_val = Path('../train_jpg')
labels = [x for x in root_val.iterdir()]
toatl_num = 0
for label in labels:
	currnt_path = label
	toatl_num += len([x for x in currnt_path.iterdir()])

print(toatl_num)


