import torch
import torchvision
import torch.nn as nn
import numpy as np
import pdb
from utils import *

def get_LL_HH(x):
	pooled = torch.nn.functional.avg_pool2d(x, 2)
	#up_pooled = torch.nn.functional.interpolate(pooled, scale_factor=2, mode='nearest')
	up_pooled = torch.nn.functional.interpolate(pooled, size=[x.size(2), x.size(3)], mode='nearest')
	HH = x - up_pooled
	LL = up_pooled
	return HH, LL

class DA_Net(nn.Module):
	def __init__(self):
		super(DA_Net, self).__init__()
		self.in_dim1 = 64
		self.in_dim2 = 128
		self.in_dim3 = 256
		self.in_dim4 = 512
		#64 * 256 * 256 => 1 * 64 * 64
		#128 * 128 * 128 => 1 * 128 * 128
		#256 * 64 * 64 => 1 * 256 * 256
		#512 * 32 * 32 => 1 * 512 * 512

		self.linear1 = nn.Sequential(
			nn.Linear(self.in_dim1**2, 1024),
			nn.ReLU(),
			#nn.LeakyReLU(0.2, inplace=True),
			#nn.Linear(1024, 100),
			#nn.LeakyReLU(0.2, inplace=True),
			#nn.Linear(100,1),
			nn.Linear(1024, 1),

		)

		self.linear2 = nn.Sequential(
			nn.Linear(self.in_dim2**2, 1024),
			nn.ReLU(),
			#nn.LeakyReLU(0.2, inplace=True),
			#nn.Linear(1024, 100),
			#nn.LeakyReLU(0.2, inplace=True),
			#nn.Linear(100,1),
			nn.Linear(1024, 1),
		)

		self.linear3 = nn.Sequential(
			nn.Linear(self.in_dim3**2, 1024),
			nn.ReLU(),
			#nn.LeakyReLU(0.2, inplace=True),
			#nn.Linear(1024, 100),
			#nn.LeakyReLU(0.2, inplace=True),
			#nn.Linear(100,1),
			nn.Linear(1024, 1),
		)

		self.linear4 = nn.Sequential(
			nn.Linear(self.in_dim4**2, 1024),
			nn.ReLU(),
			#nn.LeakyReLU(0.2, inplace=True),
			#nn.Linear(1024, 100),
			#nn.LeakyReLU(0.2, inplace=True),
			#nn.Linear(100,1),
			nn.Linear(1024, 1),
		)

	def forward(self, inputs, level=1):
		if level == 1:
			x = self.linear1(gram_matrix(inputs).view(inputs.size(0), -1))
			return x
		elif level == 2:
			x = self.linear2(gram_matrix(inputs).view(inputs.size(0), -1))
			return x
		elif level == 3:
			x = self.linear3(gram_matrix(inputs).view(inputs.size(0), -1))
			return x
		elif level == 4:
			x = self.linear4(gram_matrix(inputs).view(inputs.size(0), -1))
			return x



class DA_Net_v2(nn.Module):
	def __init__(self):
		super(DA_Net_v2, self).__init__()
		self.in_dim1 = 64
		self.in_dim2 = 128
		self.in_dim3 = 256
		self.in_dim4 = 512
		#64 * 256 * 256 => 1 * 64 * 64
		#128 * 128 * 128 => 1 * 128 * 128
		#256 * 64 * 64 => 1 * 256 * 256
		#512 * 32 * 32 => 1 * 512 * 512
		
		self.linear1 = nn.Sequential(
			nn.Linear(self.in_dim1**2, 1024),
			#nn.ReLU(),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100,1),
			#nn.Linear(1024, 1),
		)

		self.linear2 = nn.Sequential(
			nn.Linear(self.in_dim2**2, 1024),
			#nn.ReLU(),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100,1),
			#nn.Linear(1024, 1),
		)

		self.linear3 = nn.Sequential(
			nn.Linear(self.in_dim3**2, 1024),
			#nn.ReLU(),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100,1),
			#nn.Linear(1024, 1),
		)

		self.linear4 = nn.Sequential(
			nn.Linear(self.in_dim4**2, 1024),
			#nn.ReLU(),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100,1),
			#nn.Linear(1024, 1),
		)

	def forward(self, inputs, level=1):
		if level == 1:
			x = self.linear1(gram_matrix(inputs).view(inputs.size(0), -1))
			return x
		elif level == 2:
			x = self.linear2(gram_matrix(inputs).view(inputs.size(0), -1))
			return x
		elif level == 3:
			x = self.linear3(gram_matrix(inputs).view(inputs.size(0), -1))
			return x
		elif level == 4:
			x = self.linear4(gram_matrix(inputs).view(inputs.size(0), -1))
			return x


class DA_Net_v3(nn.Module):
	def __init__(self):
		super(DA_Net_v3, self).__init__()
		self.in_dim1 = 64
		self.in_dim2 = 128
		self.in_dim3 = 256
		self.in_dim4 = 512
		#64 * 256 * 256 => 1 * 64 * 64
		#128 * 128 * 128 => 1 * 128 * 128
		#256 * 64 * 64 => 1 * 256 * 256
		#512 * 32 * 32 => 1 * 512 * 512
		self.conv1 = nn.Sequential(
			nn.Conv2d(self.in_dim1, 1, kernel_size=1, bias=False),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(256//8)
		)

		self.conv2 = nn.Sequential(
			nn.Conv2d(self.in_dim2, 1, kernel_size=1, bias=False),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(128//8)
		)

		self.conv3 = nn.Sequential(
			nn.Conv2d(self.in_dim3, 1, kernel_size=1, bias=False),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(64//8)
		)

		self.conv4 = nn.Sequential(
			nn.Conv2d(self.in_dim4, 1, kernel_size=1, bias=False),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(32//8)
		)


		self.linear1 = nn.Sequential(
			nn.Linear(self.in_dim1**2 + (256//8)**2, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100,1),
		)

		self.linear2 = nn.Sequential(
			nn.Linear(self.in_dim2**2 + (128//8)**2, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100,1),
		)

		self.linear3 = nn.Sequential(
			nn.Linear(self.in_dim3**2 + (64//8)**2, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100,1),
		)

		self.linear4 = nn.Sequential(
			nn.Linear(self.in_dim4**2 + (32//8)**2, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100,1),
		)

	def forward(self, inputs, level=1):
		if level == 1:
			structure_feat = self.conv1(get_LL_HH(inputs)[0]).view(inputs.size(0), -1)
			texture_feat = gram_matrix(inputs).view(inputs.size(0), -1)
			concat_feat = torch.cat([structure_feat, texture_feat], dim=1)
			x = self.linear1(concat_feat)
			return x
		elif level == 2:
			structure_feat = self.conv2(get_LL_HH(inputs)[0]).view(inputs.size(0), -1)
			texture_feat = gram_matrix(inputs).view(inputs.size(0), -1)
			concat_feat = torch.cat([structure_feat, texture_feat], dim=1)
			x = self.linear2(concat_feat)
			return x
		elif level == 3:
			structure_feat = self.conv3(get_LL_HH(inputs)[0]).view(inputs.size(0), -1)
			texture_feat = gram_matrix(inputs).view(inputs.size(0), -1)
			concat_feat = torch.cat([structure_feat, texture_feat], dim=1)
			x = self.linear3(concat_feat)
			return x
		elif level == 4:
			structure_feat = self.conv4(get_LL_HH(inputs)[0]).view(inputs.size(0), -1)
			texture_feat = gram_matrix(inputs).view(inputs.size(0), -1)
			concat_feat = torch.cat([structure_feat, texture_feat], dim=1)
			x = self.linear4(concat_feat)
			return x

class DA_Net_v4(nn.Module):
	def __init__(self):
		super(DA_Net_v4, self).__init__()
		self.in_dim1 = 64
		self.in_dim2 = 128
		self.in_dim3 = 256
		self.in_dim4 = 512
		#64 * 256 * 256 => 1 * 64 * 64
		#128 * 128 * 128 => 1 * 128 * 128
		#256 * 64 * 64 => 1 * 256 * 256
		#512 * 32 * 32 => 1 * 512 * 512
		self.conv1 = nn.Sequential(
			nn.Conv2d(self.in_dim1, 1, kernel_size=1, bias=False),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(256//8)
		)

		self.conv2 = nn.Sequential(
			nn.Conv2d(self.in_dim2, 1, kernel_size=1, bias=False),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(128//8)
		)

		self.conv3 = nn.Sequential(
			nn.Conv2d(self.in_dim3, 1, kernel_size=1, bias=False),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(64//8)
		)

		self.conv4 = nn.Sequential(
			nn.Conv2d(self.in_dim4, 1, kernel_size=1, bias=False),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(32//8)
		)


		self.linear1 = nn.Sequential(
			nn.Linear(self.in_dim1**2 + (256//8)**2, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100,1),
		)

		self.linear2 = nn.Sequential(
			nn.Linear(self.in_dim2**2 + (128//8)**2, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100,1),
		)

		self.linear3 = nn.Sequential(
			nn.Linear(self.in_dim3**2 + (64//8)**2, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100,1),
		)

		self.linear4 = nn.Sequential(
			nn.Linear(self.in_dim4**2 + (32//8)**2, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100,1),
		)

	def forward(self, inputs, level=1):
		if level == 1:
			structure_feat = self.conv1((inputs)).view(inputs.size(0), -1)
			texture_feat = gram_matrix(inputs).view(inputs.size(0), -1)
			concat_feat = torch.cat([structure_feat, texture_feat], dim=1)
			x = self.linear1(concat_feat)
			return x
		elif level == 2:
			structure_feat = self.conv2((inputs)).view(inputs.size(0), -1)
			texture_feat = gram_matrix(inputs).view(inputs.size(0), -1)
			concat_feat = torch.cat([structure_feat, texture_feat], dim=1)
			x = self.linear2(concat_feat)
			return x
		elif level == 3:
			structure_feat = self.conv3((inputs)).view(inputs.size(0), -1)
			texture_feat = gram_matrix(inputs).view(inputs.size(0), -1)
			concat_feat = torch.cat([structure_feat, texture_feat], dim=1)
			x = self.linear3(concat_feat)
			return x
		elif level == 4:
			structure_feat = self.conv4((inputs)).view(inputs.size(0), -1)
			texture_feat = gram_matrix(inputs).view(inputs.size(0), -1)
			concat_feat = torch.cat([structure_feat, texture_feat], dim=1)
			x = self.linear4(concat_feat)
			return x

class DA_Net_v5(nn.Module):
	def __init__(self):
		super(DA_Net_v5, self).__init__()
		self.in_dim1 = 64
		self.in_dim2 = 128
		self.in_dim3 = 256
		self.in_dim4 = 512
		#64 * 256 * 256 => 1 * 64 * 64
		#128 * 128 * 128 => 1 * 128 * 128
		#256 * 64 * 64 => 1 * 256 * 256
		#512 * 32 * 32 => 1 * 512 * 512
		self.conv1 = nn.Sequential(
			nn.Conv2d(self.in_dim1, 1, kernel_size=1, bias=False),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(256//8)
		)

		self.conv2 = nn.Sequential(
			nn.Conv2d(self.in_dim2, 1, kernel_size=1, bias=False),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(128//8)
		)

		self.conv3 = nn.Sequential(
			nn.Conv2d(self.in_dim3, 1, kernel_size=1, bias=False),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(64//8)
		)

		self.conv4 = nn.Sequential(
			nn.Conv2d(self.in_dim4, 1, kernel_size=1, bias=False),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(32//8)
		)


		self.linear1 = nn.Sequential(
			nn.Linear(self.in_dim1**2 + (256//8)**2, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100, 100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100,1),
		)

		self.linear2 = nn.Sequential(
			nn.Linear(self.in_dim2**2 + (128//8)**2, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100, 100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100,1),
		)

		self.linear3 = nn.Sequential(
			nn.Linear(self.in_dim3**2 + (64//8)**2, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100, 100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100,1),
		)

		self.linear4 = nn.Sequential(
			nn.Linear(self.in_dim4**2 + (32//8)**2, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100, 100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100,1),
		)

	def forward(self, inputs, level=1):
		if level == 1:
			structure_feat = self.conv1(get_LL_HH(inputs)[0]).view(inputs.size(0), -1)
			texture_feat = gram_matrix(inputs).view(inputs.size(0), -1)
			concat_feat = torch.cat([structure_feat, texture_feat], dim=1)
			x = self.linear1(concat_feat)
			return x
		elif level == 2:
			structure_feat = self.conv2(get_LL_HH(inputs)[0]).view(inputs.size(0), -1)
			texture_feat = gram_matrix(inputs).view(inputs.size(0), -1)
			concat_feat = torch.cat([structure_feat, texture_feat], dim=1)
			x = self.linear2(concat_feat)
			return x
		elif level == 3:
			structure_feat = self.conv3(get_LL_HH(inputs)[0]).view(inputs.size(0), -1)
			texture_feat = gram_matrix(inputs).view(inputs.size(0), -1)
			concat_feat = torch.cat([structure_feat, texture_feat], dim=1)
			x = self.linear3(concat_feat)
			return x
		elif level == 4:
			structure_feat = self.conv4(get_LL_HH(inputs)[0]).view(inputs.size(0), -1)
			texture_feat = gram_matrix(inputs).view(inputs.size(0), -1)
			concat_feat = torch.cat([structure_feat, texture_feat], dim=1)
			x = self.linear4(concat_feat)
			return x

class DA_Net_v6(nn.Module):
	def __init__(self):
		super(DA_Net_v6, self).__init__()
		self.in_dim1 = 64
		self.in_dim2 = 128
		self.in_dim3 = 256
		self.in_dim4 = 512
		#64 * 256 * 256 => 1 * 64 * 64
		#128 * 128 * 128 => 1 * 128 * 128
		#256 * 64 * 64 => 1 * 256 * 256
		#512 * 32 * 32 => 1 * 512 * 512
		self.conv1 = nn.Sequential(
			nn.Conv2d(self.in_dim1, 1, kernel_size=1, bias=False),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(256//8)
		)

		self.conv2 = nn.Sequential(
			nn.Conv2d(self.in_dim2, 1, kernel_size=1, bias=False),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(128//8)
		)

		self.conv3 = nn.Sequential(
			nn.Conv2d(self.in_dim3, 1, kernel_size=1, bias=False),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(64//8)
		)

		self.conv4 = nn.Sequential(
			nn.Conv2d(self.in_dim4, 1, kernel_size=1, bias=False),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(32//8)
		)


		self.linear1 = nn.Sequential(
			nn.Linear(self.in_dim1**2 + (256//8)**2, 1024),
			nn.InstanceNorm1d(1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
			nn.InstanceNorm1d(100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100, 100),
			nn.InstanceNorm1d(100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100,1),
		)

		self.linear2 = nn.Sequential(
			nn.Linear(self.in_dim2**2 + (128//8)**2, 1024),
			nn.InstanceNorm1d(1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
			nn.InstanceNorm1d(100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100, 100),
			nn.InstanceNorm1d(100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100,1),
		)

		self.linear3 = nn.Sequential(
			nn.Linear(self.in_dim3**2 + (64//8)**2, 1024),
			nn.InstanceNorm1d(1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
			nn.InstanceNorm1d(100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100, 100),
			nn.InstanceNorm1d(100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100,1),
		)

		self.linear4 = nn.Sequential(
			nn.Linear(self.in_dim4**2 + (32//8)**2, 1024),
			nn.InstanceNorm1d(1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
			nn.InstanceNorm1d(100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100, 100),
			nn.InstanceNorm1d(100),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100,1),
		)

	def forward(self, inputs, level=1):
		if level == 1:
			structure_feat = self.conv1(get_LL_HH(inputs)[0]).view(inputs.size(0), -1)
			texture_feat = gram_matrix(inputs).view(inputs.size(0), -1)
			concat_feat = torch.cat([structure_feat, texture_feat], dim=1)
			x = self.linear1(concat_feat.unsqueeze(0)).squeeze(0)
			return x
		elif level == 2:
			structure_feat = self.conv2(get_LL_HH(inputs)[0]).view(inputs.size(0), -1)
			texture_feat = gram_matrix(inputs).view(inputs.size(0), -1)
			concat_feat = torch.cat([structure_feat, texture_feat], dim=1)
			x = self.linear2(concat_feat.unsqueeze(0)).squeeze(0)
			return x
		elif level == 3:
			structure_feat = self.conv3(get_LL_HH(inputs)[0]).view(inputs.size(0), -1)
			texture_feat = gram_matrix(inputs).view(inputs.size(0), -1)
			concat_feat = torch.cat([structure_feat, texture_feat], dim=1)
			x = self.linear3(concat_feat.unsqueeze(0)).squeeze(0)
			return x
		elif level == 4:
			structure_feat = self.conv4(get_LL_HH(inputs)[0]).view(inputs.size(0), -1)
			texture_feat = gram_matrix(inputs).view(inputs.size(0), -1)
			concat_feat = torch.cat([structure_feat, texture_feat], dim=1)
			x = self.linear4(concat_feat.unsqueeze(0)).squeeze(0)
			return x


class New_DA_Net_v1(nn.Module):
	def __init__(self, size=256):
		super(New_DA_Net_v1, self).__init__()
		self.in_dim1 = 64
		self.in_dim2 = 128
		self.in_dim3 = 256
		self.input_size = size
		#64 * 256 * 256 => 1 * 64 * 64
		#128 * 128 * 128 => 1 * 128 * 128
		#256 * 64 * 64 => 1 * 256 * 256
		
		self.conv1 = nn.Sequential(
			nn.Conv2d(self.in_dim1, 64, kernel_size=1, bias=False),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(64)
		)

		self.conv2 = nn.Sequential(
			nn.Conv2d(self.in_dim2, 64, kernel_size=1, bias=False),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(64)
		)

		self.conv3 = nn.Sequential(
			nn.Conv2d(self.in_dim3, 64, kernel_size=1, bias=False),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(64)
		)


		self.linear1 = nn.Sequential(
			nn.Linear(self.in_dim1**2, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
		)

		self.linear2 = nn.Sequential(
			nn.Linear(self.in_dim2**2, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
		)

		self.linear3 = nn.Sequential(
			nn.Linear(self.in_dim3**2, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
		)

		self.shared_conv = nn.Sequential(
			# 164 x 64 x 64
			nn.Conv2d(100+64, 256, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),

			# 256 x 32 x 32
			nn.Conv2d(256, 256, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),

			# 256 x 16 x 16
			nn.Conv2d(256, 512, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),

			# 512 x 8 x 8
			nn.Conv2d(512, 512, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),

			# 512 x 4 x 4
			nn.Conv2d(512, 1, 4, 1, 0, bias=False),
		)


	def forward(self, inputs, level=1):
		if level == 1:
			texture_feat = self.linear1(gram_matrix(inputs).view(inputs.size(0), -1)).unsqueeze(2).unsqueeze(3).repeat(1, 1, 64, 64)
			structure_feat = self.conv1(inputs)
			concat_feat = torch.cat([structure_feat, texture_feat], dim=1)
			x = self.shared_conv(concat_feat).squeeze()
			return x
		elif level == 2:
			texture_feat = self.linear2(gram_matrix(inputs).view(inputs.size(0), -1)).unsqueeze(2).unsqueeze(3).repeat(1, 1, 64, 64)
			structure_feat = self.conv2(inputs)
			concat_feat = torch.cat([structure_feat, texture_feat], dim=1)
			x = self.shared_conv(concat_feat).squeeze()
			return x
		elif level == 3:
			texture_feat = self.linear3(gram_matrix(inputs).view(inputs.size(0), -1)).unsqueeze(2).unsqueeze(3).repeat(1, 1, 64, 64)
			structure_feat = self.conv3(inputs)
			concat_feat = torch.cat([structure_feat, texture_feat], dim=1)
			x = self.shared_conv(concat_feat).squeeze()
			return x

class New_DA_Net_v2(nn.Module):
	def __init__(self, size=256):
		super(New_DA_Net_v2, self).__init__()
		self.in_dim1 = 64
		self.in_dim2 = 128
		self.in_dim3 = 256
		self.input_size = size
		#64 * 256 * 256 => 1 * 64 * 64
		#128 * 128 * 128 => 1 * 128 * 128
		#256 * 64 * 64 => 1 * 256 * 256
		
		self.conv1 = nn.Sequential(
			nn.Conv2d(self.in_dim1, 64, kernel_size=1, bias=False),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(64)
		)

		self.conv2 = nn.Sequential(
			nn.Conv2d(self.in_dim2, 64, kernel_size=1, bias=False),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(64)
		)

		self.conv3 = nn.Sequential(
			nn.Conv2d(self.in_dim3, 64, kernel_size=1, bias=False),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(64)
		)


		self.linear1 = nn.Sequential(
			nn.Linear(self.in_dim1**2, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
		)

		self.linear2 = nn.Sequential(
			nn.Linear(self.in_dim2**2, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
		)

		self.linear3 = nn.Sequential(
			nn.Linear(self.in_dim3**2, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
		)

		self.shared_conv = nn.Sequential(
			# 164 x 64 x 64
			nn.Conv2d((100+64)*3, 256, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),

			# 256 x 32 x 32
			nn.Conv2d(256, 256, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),

			# 256 x 16 x 16
			nn.Conv2d(256, 512, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),

			# 512 x 8 x 8
			nn.Conv2d(512, 512, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),

			# 512 x 4 x 4
			nn.Conv2d(512, 1, 4, 1, 0, bias=False),
			#nn.Conv2d(512, 1, 4, 4, 0, bias=False) -> patchGAN style이네 지금은... 이걸로 바꿔야 vanilla GAN
		)


	def forward(self, inputs, level=1):
		in1, in2, in3 = inputs[0], inputs[1], inputs[2]
		texture_feat_1 = self.linear1(gram_matrix(in1).view(in1.size(0), -1)).unsqueeze(2).unsqueeze(3).repeat(1, 1, 64, 64)
		structure_feat_1 = self.conv1(in1)
		concat_feat_1 = torch.cat([structure_feat_1, texture_feat_1], dim=1)

		texture_feat_2 = self.linear2(gram_matrix(in2).view(in2.size(0), -1)).unsqueeze(2).unsqueeze(3).repeat(1, 1, 64, 64)
		structure_feat_2 = self.conv2(in2)
		concat_feat_2 = torch.cat([structure_feat_2, texture_feat_2], dim=1)

		texture_feat_3 = self.linear3(gram_matrix(in3).view(in3.size(0), -1)).unsqueeze(2).unsqueeze(3).repeat(1, 1, 64, 64)
		structure_feat_3 = self.conv3(in3)
		concat_feat_3 = torch.cat([structure_feat_3, texture_feat_3], dim=1)
		
		total_feat= torch.cat([concat_feat_1, concat_feat_2, concat_feat_3], dim=1)
		x = self.shared_conv(total_feat)
		return x


class New_DA_Net_v3(nn.Module):
	def __init__(self, size=256):
		super(New_DA_Net_v3, self).__init__()
		self.in_dim1 = 64
		self.in_dim2 = 128
		self.in_dim3 = 256
		self.input_size = size
		#64 * 512 * 512 => 1 * 64 * 64
		#128 * 256 * 256 => 1 * 128 * 128
		#256 * 128 * 128 => 1 * 256 * 256
		
		self.linear1 = nn.Sequential(
			nn.Linear(self.in_dim1**2, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
		)

		self.linear2 = nn.Sequential(
			nn.Linear(self.in_dim2**2, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
		)

		self.linear3 = nn.Sequential(
			nn.Linear(self.in_dim3**2, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 100),
		)

		self.conv1 = nn.Sequential(
			# 164 x 512 x 512
			nn.Conv2d(100+64, 256, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),

			# 256 x 256 x 256
			nn.Conv2d(256, 256, 4, 4, 0, bias=False),
			nn.LeakyReLU(0.2, inplace=True),

			# 256 x 64 x 64
			nn.Conv2d(256, 512, 4, 4, 0, bias=False),
			nn.LeakyReLU(0.2, inplace=True),

			# 512 x 16 x 16
			nn.Conv2d(512, 512, 4, 4, 0, bias=False),
			nn.LeakyReLU(0.2, inplace=True),

			# 512 x 4 x 4
			nn.Conv2d(512, 1, 4, 4, 0, bias=False),
		)

		self.conv2 = nn.Sequential(
			# 228 x 256 x 256
			nn.Conv2d(100+128, 256, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),

			# 256 x 128 x 128
			nn.Conv2d(256, 256, 4, 4, 0, bias=False),
			nn.LeakyReLU(0.2, inplace=True),

			# 256 x 32 x 32
			nn.Conv2d(256, 512, 4, 4, 0, bias=False),
			nn.LeakyReLU(0.2, inplace=True),

			# 512 x 8 x 8
			nn.Conv2d(512, 512, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),

			# 512 x 4 x 4
			nn.Conv2d(512, 1, 4, 4, 0, bias=False),
		)

		self.conv3 = nn.Sequential(
			# 356 x 128 x 128
			nn.Conv2d(100+256, 400, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),

			# 400 x 64 x 64
			nn.Conv2d(400, 512, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),

			# 512 x 32 x 32
			nn.Conv2d(512, 512, 4, 4, 0, bias=False),
			nn.LeakyReLU(0.2, inplace=True),

			# 512 x 8 x 8
			nn.Conv2d(512, 512, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),

			# 512 x 4 x 4
			nn.Conv2d(512, 1, 4, 4, 0, bias=False),
		)


	def forward(self, inputs, level=1):
		w, h = inputs.size(2), inputs.size(3)
		if level == 1:
			texture_feat = self.linear1(gram_matrix(inputs).view(inputs.size(0), -1)).unsqueeze(2).unsqueeze(3).repeat(1, 1, w, h)
			structure_feat = inputs
			concat_feat = torch.cat([structure_feat, texture_feat], dim=1)
			x = self.conv1(concat_feat).squeeze()
			return x
		elif level == 2:
			texture_feat = self.linear2(gram_matrix(inputs).view(inputs.size(0), -1)).unsqueeze(2).unsqueeze(3).repeat(1, 1, w, h)
			structure_feat = inputs
			concat_feat = torch.cat([structure_feat, texture_feat], dim=1)
			x = self.conv2(concat_feat).squeeze()
			return x
		elif level == 3:
			texture_feat = self.linear3(gram_matrix(inputs).view(inputs.size(0), -1)).unsqueeze(2).unsqueeze(3).repeat(1, 1, w, h)
			structure_feat = inputs
			concat_feat = torch.cat([structure_feat, texture_feat], dim=1)
			x = self.conv3(concat_feat).squeeze()
			return x