import os

origin_dir = './kinetics-400/'
file_list = os.listdir(origin_dir)
f_basename = 'kinetics-400.tar.gz.part{}-{}'
print(len(file_list))
nn = 0

target_fname = './kinetics-400-source.tar.gz'
w_f = open(target_fname,'wb')

for N in range(1,len(file_list)//4+1):
	for i in range(4):
		file_name = f_basename.format(N,i)                       # 文件名

		print(file_name,os.path.exists(origin_dir+file_name))   #判断每个文件是否存在
		src_f = open(origin_dir+file_name,'rb')                #以读的方式打开每个文件
		w_f.write(src_f.read())                                # 读取原文件后写入目标文件
		w_f.flush()
		src_f.close()
		nn+=1

w_f.close()
print(nn)
