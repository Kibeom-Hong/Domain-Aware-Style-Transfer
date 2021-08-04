import os, sys, random, cv2, pdb, csv

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import imageio
import numpy as np
import scipy.misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skvideo.io
from PIL import Image
import natsort
Image.MAX_IMAGE_PIXELS = 1000000000



class MSCOCO(torch.utils.data.Dataset):
	def __init__(self, root_path, imsize=None, cropsize=None, cencrop=False):
		super(MSCOCO, self).__init__()

		#self.file_names = sorted(os.listdir(root_path))
		self.root_path = root_path
		self.file_names = sorted([os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(os.path.join(self.root_path,'train2017')) for f in files if f.endswith('jpg') or f.endswith('png')])
		self.file_names += sorted([os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(os.path.join(self.root_path,'val2017')) for f in files if f.endswith('jpg') or f.endswith('png')])
		self.transform = _transformer(imsize, cropsize, cencrop)

	def __len__(self):
		return len(self.file_names)

	def __getitem__(self, index):
		#image = Image.open(os.path.join(self.root_path + self.file_names[index])).convert("RGB")
		try:
			image = Image.open(self.file_names[index]).convert("RGB")
		except:
			print(self.file_names[index])

		return self.transform(image)

class WiKiART(torch.utils.data.Dataset):
	def __init__(self, root_path, imsize=None, cropsize=None, cencrop=False):
		super(WiKiART, self).__init__()

		#self.file_names = sorted(os.listdir(root_path))
		#self.file_names = sorted([os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(self.root_path) for f in files if f.endswith('jpg') or f.endswith('png')])
		self.root_path = root_path
		self.file_names = []
		self.transform = _transformer(imsize, cropsize, cencrop)
		art_path = '../../dataset/wikiart_csv'
		self.csv_files = sorted([os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(art_path) for f in files if (f.split('_')[-1]).split('.')[0] == 'train' ]) 
		for csv_file in self.csv_files:
			f = open(csv_file, 'r', encoding='utf-8')
			rdr = csv.reader(f)
			for line in rdr:
				self.file_names.append(os.path.join(self.root_path, line[0]))

	def __len__(self):
		return len(self.file_names)

	def __getitem__(self, index):
		#image = Image.open(os.path.join(self.root_path + self.file_names[index])).convert("RGB")
		try:
			image = Image.open(self.file_names[index]).convert("RGB")
		except:
			print(self.file_names[index])
		return self.transform(image)

class TestDataset(torch.utils.data.Dataset):
	def __init__(self, imsize=None, cropsize=None, cencrop=False):
		super(TestDataset, self).__init__()

		self.transform = _transformer(imsize, cropsize, cencrop)

		photo_path = '../../dataset/MSCoCo'
		self.photo_file_names = sorted([os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(os.path.join(photo_path, 'test2017')) for f in files if f.endswith('jpg') or f.endswith('png') or f.endswith('jpeg') ])
		

		art_root_path = '../../dataset/wikiart'
		self.art_file_names = []
		art_path = '../../dataset/wikiart_csv'
		self.csv_files = sorted([os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(art_path) for f in files if (f.split('_')[-1]).split('.')[0] == 'val' ]) 
		for csv_file in self.csv_files:
			f = open(csv_file, 'r', encoding='utf-8')
			rdr = csv.reader(f)
			for line in rdr:
				self.art_file_names.append(os.path.join(art_root_path, line[0]))
		
	def __len__(self):
		return len(self.photo_file_names)

	def __getitem__(self, index):
		#image = Image.open(os.path.join(self.root_path + self.file_names[index])).convert("RGB")
		try:
			photo_image = Image.open(self.photo_file_names[index]).convert("RGB")
			art_image = Image.open(self.art_file_names[index]).convert("RGB")
		except:
			print(self.photo_file_names[index])
			print(self.art_file_names[index])
		return self.transform(photo_image), self.transform(art_image)


class Art_Transfer_TestDataset(torch.utils.data.Dataset):
	def __init__(self, root_path, imsize=None, cropsize=None, cencrop=False):
		super(Art_Transfer_TestDataset, self).__init__()

		self.transform = _transformer()
		art_root_path = '../../dataset/wikiart'
		self.art_file_names = []
		art_path = '../../dataset/wikiart_csv'
		self.csv_files = sorted([os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(art_path) for f in files if (f.split('_')[-1]).split('.')[0] == 'val' ]) 
		for csv_file in self.csv_files:
			f = open(csv_file, 'r', encoding='utf-8')
			rdr = csv.reader(f)
			for line in rdr:
				self.art_file_names.append(os.path.join(art_root_path, line[0]))
		
	def __len__(self):
		return len(self.art_file_names)

	def __getitem__(self, index):
		#image = Image.open(os.path.join(self.root_path + self.file_names[index])).convert("RGB")
		try:
			art_image = Image.open(self.art_file_names[index]).convert("RGB")
		except:
			print(self.art_file_names[index])
		return self.transform(art_image)

class Transfer_TestDataset(torch.utils.data.Dataset):
	def __init__(self, root_path, imsize=None, cropsize=None, cencrop=False, type='photo', is_test=False):
		super(Transfer_TestDataset, self).__init__()

		#self.file_names = sorted(os.listdir(root_path))
		self.root_path = root_path
		if is_test:
			self.transform = _transformer()#_transformer(imsize, cropsize, cencrop)#_transformer()# #여기도 나중에 코드정리할 때 분리해주기
			#self.transform = _transformer(imsize)#_transformer()# #여기도 나중에 코드정리할 때 분리해주기
		else:
			self.transform = _transformer(imsize, cropsize, cencrop)#_transformer()# #여기도 나중에 코드정리할 때 분리해주기
		
		if type =='photo':
			self.file_names = (sorted([os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(self.root_path) for f in files if f.endswith('jpg') or f.endswith('png') or f.endswith('JPG') or f.endswith('jpeg')]))
		else:
			self.file_names = natsort.natsorted(sorted([os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(self.root_path) for f in files if f.endswith('jpg') or f.endswith('png') or f.endswith('JPG') or f.endswith('jpeg')]))
		
	def __len__(self):
		return len(self.file_names)

	def __getitem__(self, index):
		#image = Image.open(os.path.join(self.root_path + self.file_names[index])).convert("RGB")
		try:
			image = Image.open(self.file_names[index]).convert("RGB")
		except:
			print(self.file_names[index])
			image = Image.open(self.file_names[index-1]).convert("RGB")
		return self.transform(image)
		

class Transfer_Video_TestDataset(torch.utils.data.Dataset):
	def __init__(self, root_path, imsize=None, cropsize=None, cencrop=False, T=16):
		super(Transfer_Video_TestDataset, self).__init__()

		#self.file_names = sorted(os.listdir(root_path))
		self.T = T
		self.root_path = root_path
		self.transform = _transformer(imsize, cropsize, cencrop)
		self.file_names = (sorted([os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(self.root_path) for f in files if f.endswith('mp4') or f.endswith('avi')]))


	def __len__(self):
		return len(self.file_names)

	def trim(self, video):
		if video.shape[1] > self.T:
			start = np.random.randint(0, video.shape[1] - (self.T*1) + 1)
			end = start + self.T
			return video[:, start:end, :, :]
		else:
			index = ((video.shape[1] / self.T)*np.arange(self.T)).astype(np.int32)
			return video[:, index, :, :]

	def video_transform(self, video):
		vid = []
		for frame_idx in range(video.shape[0]):
			frame = self.transform(Image.fromarray(video[frame_idx,:,:,:], 'RGB'))
			vid.append(frame)
		vid = torch.stack(vid).permute(1,0,2,3)
		return vid

	def __getitem__(self, index):
		video_path = self.file_names[index]
		try:
			video = skvideo.io.vread(video_path)
			video = self.video_transform(video)
			video = self.trim(video)
		except:
			print(self.file_names[index])
		return video

def lastest_arverage_value(values, length=100):
	if len(values) < length:
		length = len(values)
	return sum(values[-length:])/length


def _normalizer(denormalize=False):
	# set Mean and Std of RGB channels of IMAGENET to use pre-trained VGG net
	MEAN = [0.485, 0.456, 0.406]
	STD = [0.229, 0.224, 0.225]    
	
	if denormalize:
		MEAN = [-mean/std for mean, std in zip(MEAN, STD)]
		STD = [1/std for std in STD]
	
	return transforms.Normalize(mean=MEAN, std=STD)

def _transformer(imsize=None, cropsize=None, cencrop=False):
	normalize = _normalizer()
	transformer = []
	w, h = imsize, imsize
	if imsize:
		transformer.append(transforms.Resize(imsize))
	if cropsize:
		if cencrop:
			transformer.append(transforms.CenterCrop(cropsize))
		else:
			transformer.append(transforms.RandomCrop(cropsize))

	transformer.append(transforms.ToTensor())
	transformer.append(normalize)
	return transforms.Compose(transformer)

def imsave(tensor, path, nrow=4, npadding=0):
	denormalize = _normalizer(denormalize=True)
	if tensor.is_cuda:
		tensor = tensor.cpu()
	tensor = torchvision.utils.make_grid(tensor, nrow=nrow, padding=npadding)
	torchvision.utils.save_image(denormalize(tensor).clamp_(0.0, 1.0), path)
	return None

def denorm(tensor, nrow=4, npadding=0):
	denormalize = _normalizer(denormalize=True)
	if tensor.is_cuda:
		tensor = tensor.cpu()
	tensor = torchvision.utils.make_grid(tensor, nrow=nrow, padding=npadding)
	return (denormalize(tensor).clamp_(0.0, 1.0))

def imload(path, imsize=None, cropsize=None, cencrop=False):
	transformer = _transformer(imsize, cropsize, cencrop)
	return transformer(Image.open(path).convert("RGB")).unsqueeze(0)

def imshow(tensor):
	denormalize = _normalizer(denormalize=True)    
	if tensor.is_cuda:
		tensor = tensor.cpu()    
	tensor = torchvision.utils.make_grid(denormalize(tensor.squeeze(0)))
	image = transforms.functional.to_pil_image(tensor.clamp_(0.0, 1.0))
	return image

def maskload(path):
	mask = Image.open(path).convert('L')
	return transforms.functional.to_tensor(mask).unsqueeze(0)

