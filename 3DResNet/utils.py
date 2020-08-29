import h5py


# 得到每个视频的帧数量
def get_n_frames(video_path):
    return len([
        x for x in video_path.iterdir()     # 在video子目录中循环
        if 'image' in x.name and x.name[0] != '.'
    ])

def get_n_frames_hdf5(video_path):
    with h5py.File(video_path,'r') as f:
        video_data = f['video']

        return len(video_data)

