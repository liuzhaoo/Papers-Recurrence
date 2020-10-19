import os

tar = './train_256/'
classes = os.listdir(tar)
n = 0
for cl_name in classes:
	video_name = os.listdir(tar+cl_name)
	for videos in video_name:

		# print(videos)
		# ext0 = videos.split('.')[0]
		# new = tar+cl_name+'/'+ ext0+'.mp4'
		# old = tar+cl_name+'/'+ videos
		# os.rename(old,new)
		# ext2 = videos.split('.')[2]
		if videos[-3:] != 'mp4' :
			n += 1
		# print(ext0)
			

print(n)
	

