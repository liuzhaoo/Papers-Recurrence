def get_mean_std(value_scale, dataset):
	assert dataset in ['activitynet', 'kinetics', '0.5']

	if dataset == 'activitynet':
		mean = [0.4477, 0.4209, 0.3906]
		std = [0.2767, 0.2695, 0.2714]
	elif dataset == 'kinetics':
		mean = [0.4345, 0.4051, 0.3775]
		std = [0.2768, 0.2713, 0.2737]
	elif dataset == '0.5':
		mean = [0.5, 0.5, 0.5]
		std = [0.5, 0.5, 0.5]
	elif dataset == 'ucf101':
		mean = [0.39755231, 0.38231853, 0.35171264]
		std = [0.24180283, 0.23505084, 0.23117465]

	mean = [x * value_scale for x in mean]
	std = [x * value_scale for x in std]

	return mean, std
