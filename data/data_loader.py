import os
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import torch
from .dataset_util import *
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import random_split

def get_loader(image_dir, crop_size=0, image_size=0, batch_size=16, normalize=True, noise=True, split='train', num_workers=2):
	transform = []
	if split == 'train':
		transform.append(transforms.RandomHorizontalFlip())
	if crop_size > 0:
		transform.append(transforms.CenterCrop([crop_size, crop_size]))
	if image_size > 0:
		transform.append(transforms.Resize([image_size, image_size]))
	transform.append(transforms.ToTensor())
	if normalize:
		transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
	if noise:
		transform.append(transforms.Lambda(lambda x: x + 1./128 * torch.randn(x.size())))
	# TODO: add mirror augmentation (stylegan)
	# TODO: (generaory) change down / up sampling method (stylegan)
	transform = transforms.Compose(transform)

	dataset = datasets.ImageFolder(image_dir, transform=transform)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=num_workers)
	return data_loader

def get_loaders(dir_dataset, crop_size=0, image_size=0, batch_size=16, num_workers=2):
	modes = ['train', 'val', 'test']
	loaders = {}
	for mode in modes:
		dir_imagefolder = os.path.join(dir_dataset, mode)
		if not os.path.exists(dir_imagefolder):
			continue
		loader = get_loader(dir_imagefolder, crop_size, image_size, batch_size, mode, num_workers)
		loaders[mode] = loader
	return loaders
