from pathlib import Path
from PIL import Image
import cv2
import torch
import numpy as np

root_path = Path('/home/lzhao/FILE/datasets/UCF101/ucf101')

labels = [x.stem for x in root_path.iterdir()]

labelnum = len(labels)
curnum = 0
means = [0, 0, 0]
stdevs = [0, 0, 0]
num_imgs = 0
for label in labels:
	current_path = root_path / label

	videos = [x.stem for x in current_path.iterdir()]
	videos = videos[::5]
	curnum += 1
	for video in videos:
		video_path = current_path / video

		pics = [x.stem for x in video_path.iterdir()]

		pics = pics[::7]
		print('正在处理 {} 类（{}/{}）的 {} 视频'.format(label, curnum, labelnum, video))
		print('.....')
		for pic in pics:
			pic += '.jpg'
			pic_path = video_path / pic

			img = cv2.imread(str(pic_path))

			img = np.asarray(img).astype(np.float32) / 255
			num_imgs += 1
			for i in range(3):
				means[i] += img[:, :, i].mean()  # i=0,1,2 分别计算图像的BGR维度mean，放到0，1，2维度
				stdevs[i] += img[:, :, i].std()

print(num_imgs)
means.reverse()
stdevs.reverse()


means = np.asarray(means) / num_imgs
stdevs = np.asarray(stdevs) / num_imgs

print('Mean = {}'.format(means))
print('Std = {}'.format(stdevs))

#Mean = [0.39755231 0.38231853 0.35171264]
#Std = [0.24180283 0.23505084 0.23117465]