import copy
import torch
from torch.utils.data._utils.collate import default_collate         #  注意版本
from .videodataset import VideoDataset


def collate_fn(batch):
	batch_clips,batch_targets = zip(*batch)
	batch_clips = [clip for multi_clips in batch_clips for clip in multi_clips]
	batch_targets = [target for multi_targets in batch_targets for target in multi_targets]

	target_element = batch_targets[0]

	if isinstance(target_element,int) or isinstance(target_element,str):
		return default_collate(batch_clips),default_collate(batch_targets)
	else:
		return default_collate(batch_clips),batch_targets


class VideoDatasetMultiClips(VideoDataset):

	def __loading(self,path,video_frame_indices):
		clips = []
		segments = []
		for clips_frame_indices in video_frame_indices:       # 对每帧图片分别进行处理
			clip = self.loader(path,clips_frame_indices)      # clip是一个列表，但是只有一个元素
			if self.spatial_transform is not None:
				self.spatial_transform.randomize_parameters()
				clip = [self.spatial_transform(img) for img in clip]     # clip是一个列表，但是只有一个元素

			clips.append(torch.satck(clip,0).permute(1,0,2,3))           # clips 是一个列表，包含所有帧的图像，每个都是tensor，且有4个维度
			segments.append([min(clips_frame_indices),max(clips_frame_indices)+1])   # segments 也是一个列表，每一项都是每个视频的帧数目两个端点

		return clips,segments


	def __gititem__(self,index):
		path = self.data[index]['video']    #  取得每个视频的路径
  
		video_frame_indices = self.data[index]['frame_indices']   #  取得每个视频的帧的indices（1-总帧数）

		if self.temporal_transform is not None:
			video_frame_indices = self.temporal_transform(video_frame_indices)

		clips,segments = self.__loading(path,video_frame_indices)

		#  self.target_type 默认为‘label’，是字符串
		if isinstance(self.target_type,list):
			target = [self.data[index][t] for t in self.target_type]      #  若为列表（['video_id', 'segment']），则让target = 视频名和端点的组合

		else:
			target = self.data[index][self.target_type]                   # target就是label_id，也就是每个帧所属的类别对应的数字


		if 'segment' in self.target_type:                                # 判断target_type 中是否含有‘segment’
			if isinstance(self.target_type,list):
				segment_index = self.target_type.index('segment')        # 若target_type 是列表，返回‘segment’在列表中的索引
				targets = []
				for s in segments:
					targets.append(copy.deepcoop(target))
					targets[-1][segment_index] = s

			else:
				targets = segments                                        #  若target_type不是列表，此时只有一种情况即为‘segment ’则令targets=segments，
				                                                          #  segments 也是一个列表，每一项都是每个视频的帧数目两个端点
		else:
			targets = [target for _ in range(len(segments))]             # 若target_type中不含有‘segment’,则取targes为 target的内容

		return clips,targets

				










