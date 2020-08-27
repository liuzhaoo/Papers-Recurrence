import subprocess
import argparse
from pathlib import Path

from joblib import Parallel,delayed
x = './testvideo'
def video_process(video_file_path,dst_root_path,ext,fps=-1,size = 240):
    if ext!= video_file_path.suffix:
        return


    ffprobe_cmd = ('ffprobe -v error -select_streams v:0 '
               '-of default=noprint_wrappers=1:nokey=1 -show_entries '
               'stream=width,height,avg_frame_rate,duration').split()

    ffprobe_cmd.append(str(video_file_path))

    p = subprocess.run(ffprobe_cmd,capture_output=True)           # 将p作为子进程
    res = p.stdout.decode('utf-8').splitlines()                  # 将此进程的标准输出进行编码，并按行分开,返回一个列表

    if len(res) < 4:
    	return 

    frame_rate = [float(r) for r in res[2].split('/')]          #   帧率
    frame_rate = frame_rate[0] / frame_rate[1]                  #   进一步计算

    duration = float(res[3])                                    # 视频持续时间
    n_frames = int(frame_rate*duration)                         # 帧数

    name = video_file_path.stem                                 # 此路径下最后一级的名称，不包含后缀，这里得到的是每个视频的名称

    dst_dir_path = dst_root_path / name                         # 将每个视频的名称接到数据集根目录后面























