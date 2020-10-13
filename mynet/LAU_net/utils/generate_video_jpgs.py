import subprocess
import argparse
from pathlib import Path

from joblib import Parallel, delayed


def video_process(video_file_path, dst_root_path, ext, fps=-1, size=240):
    if ext != video_file_path.suffix:
        return

    ffprobe_cmd = ('ffprobe -v error -select_streams v:0 '
                   '-of default=noprint_wrappers=1:nokey=1 -show_entries '
                   'stream=width,height,avg_frame_rate,duration').split()  # 获取视频的宽，高，平均帧率（每秒的帧数），持续时间

    ffprobe_cmd.append(str(video_file_path))

    p = subprocess.run(ffprobe_cmd, capture_output=True)  # 将p作为子进程
    res = p.stdout.decode('utf-8').splitlines()  # 将此进程的标准输出进行编码，并按行分开,返回一个列表

    if len(res) < 4:
        return
    #   帧率，ffprobe计算得到的帧率是n帧/s的格式，这里将分号前后的两个数分别取出并转换为浮点数后放到列表里
    frame_rate = [float(r) for r in res[2].split('/')]
    frame_rate = frame_rate[0] / frame_rate[1]  # 得到帧率

    if res[3] == 'N/A':
        return
    duration = float(res[3])  # 视频持续时间
    n_frames = int(frame_rate * duration)  # 得到视频的总帧数

    name = video_file_path.stem  # 此路径下最后一级的名称，不包含后缀，这里得到的是每个视频的名称
    dst_dir_path = dst_root_path / name  # 将每个视频的名称接到数据集根目录后面,这里的数据集根目录其实是传入了每个类的目录
    dst_dir_path.mkdir(exist_ok=True)  # 在根目录下新建以视频名为名的文件夹
    n_exist_frames = len([
        x for x in dst_dir_path.iterdir()
        if x.suffix == '.jpg' and x.name[0] != '.'
    ])

    if n_exist_frames >= n_frames:
        return

    width = int(res[0])
    height = int(res[1])
    # 如果宽大于高，令此参数为-1：240，在转换分辨率时，将较小的一边设为指定的分辨率大小，另一边原来的关系缩放
    if width > height:
        vf_param = 'scale=-1:{}'.format(size)
    else:
        vf_param = 'scale={}:-1'.format(size)

    if fps > 0:
        vf_param += ',minterpolate={}'.format(fps)

    ffmpeg_cmd = ['ffmpeg', '-i', str(video_file_path), '-vf', vf_param]
    ffmpeg_cmd += ['-threads', '1', '{}/image_%05d.jpg'.format(dst_dir_path)]
    print(ffmpeg_cmd)
    subprocess.run(ffmpeg_cmd)
    print('\n')


def class_process(class_dir_path, dst_root_path, ext, fps=-1, size=240):
    if not class_dir_path.is_dir():
        return

    dst_class_path = dst_root_path / class_dir_path.name  # 类的路径的等于根路径接上传入的类的路径
    dst_class_path.mkdir(exist_ok=True)

    # 在每个类的文件夹下的文件（所有的视频文件）进行迭代，video_file_path就是每个视频文件的路径
    for video_file_path in sorted(class_dir_path.iterdir()):
        video_process(video_file_path, dst_class_path, ext, fps, size)


if __name__ == '__main__':
    input_dir = '/home/lzhao/文档/datasets/kinetics/train_256'
    target_dir = '/home/lzhao/文档/datasets/kinetics/train_jpg'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dir_path', default=input_dir, type=Path, help='视频文件的路径')
    parser.add_argument(
        '--dst_path', default=target_dir, type=Path, help='得到的jpg文件路径')
    parser.add_argument(
        '--n_jobs', default=-1, type=int, help='并行工作的数量')
    parser.add_argument(
        '--fps', default=-1, type=int, help='输出视频的帧率，-1代表原始帧率')
    parser.add_argument(
        '--size', default=240, type=int, help='输出视频的帧大小')

    args = parser.parse_args()

    # args.dir_path = input_dir
    # args.dst_path = target_dir

    ext = '.mp4'

    class_dir_paths = [x for x in sorted(args.dir_path.iterdir())]  # 取所有类别名，放到列表中
    # test_set_video_paths = args.dir_path/'test'                    在视频路径下添加一个测试集路径

    # if test_set_video_path.exists():
    # 	class_dir_paths.append(test_set_video_path)

    # class_dir_paths 是一个路径的列表，每一项是一个类的路径（原数据集）
    # 在class_dir_paths中迭代得到class_dir_path和dst_path(生成的数据集根目录)传入class_process()函数，再根据此两个路径
    # 组合得到生成数据集的每个类目录dst_class_path
    # 在class_process() 函数中，对class_dir_path的子目录进行遍历，得到每个视频的路径video_file_path
    # 将video_file_path和dst_class_path传入vidoe_process()函数,取每个视频的信息进行处理，保存在dst_class_path和视频名组合的文件夹下
    status_list = Parallel(
        n_jobs=args.n_jobs,
        backend='threading')(delayed(class_process)(
        class_dir_path, args.dst_path, ext, args.fps, args.size)
                             for class_dir_path in class_dir_paths)
