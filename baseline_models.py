import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import numpy as np
import math
from utils import *
from style_decorator import StyleDecorator

def size_arrange(x):
	x_w, x_h = x.size(2), x.size(3)

	if (x_w%2) != 0:
		x_w = (x_w//2)*2
	if (x_h%2) != 0:
		x_h = (x_h//2)*2

	if ( x_h > 1024 ) or (x_w > 1024) :
		old_x_w = x_w
		x_w = x_w//2
		x_h = int(x_h*x_w/old_x_w)
	
	return F.interpolate(x, size=(x_w, x_h))

def get_LL_HH(x):
	pooled = torch.nn.functional.avg_pool2d(x, 2)
	up_pooled = torch.nn.functional.interpolate(pooled, size=[x.size(2), x.size(3)], mode='nearest')
	HH = x - up_pooled
	LL = up_pooled
	return HH, LL

def gaussian_blur(x, alpha):
		B, C, W, H = x.size()
		results = []
		
		if B > 1:
			alpha = alpha.squeeze()

		for batch in range(B):
			#kernel_size = int(int(int(alpha[batch].item()*10)/2)*2)+1
			kernel_size = int(int(int(alpha[batch].item()*8)/2)*2)+1
			#kernel_size = int(int(int(alpha[batch].item()*6)/2)*2)+1
			

			channels = C
			x_cord = torch.arange(kernel_size+0.)
			x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
			y_grid = x_grid.t()
			xy_grid = torch.stack([x_grid, y_grid], dim=-1)
			mean = (kernel_size - 1)//2
			diff = -torch.sum((xy_grid - mean)**2., dim=-1)
			gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
										kernel_size=kernel_size, groups=channels, bias=False)

			gaussian_filter.weight.requires_grad = False

			sigma = 16
			variance = sigma**2.
			gaussian_kernel = (1./(2.*math.pi*variance)) * torch.exp(diff /(2*variance))
			gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
			gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
			gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
			gaussian_kernel = gaussian_kernel.cuda()
			gaussian_filter.weight.data = gaussian_kernel
			output = gaussian_filter(torch.nn.functional.pad(x[batch].unsqueeze(0), (mean ,mean, mean, mean), "replicate"))

			results.append(output)
		return  torch.stack(results).squeeze(1)

class Noise(nn.Module):
	def __init__(self, use_noise, sigma=0.2):
		super(Noise, self).__init__()
		self.use_noise = use_noise
		self.sigma = sigma

	def forward(self, x):
		if self.use_noise:
			return x + self.sigma * torch.autograd.Variable(torch.cuda.FloatTensor(x.size()).normal_(), requires_grad=False)
		else:
			return x


class AdaIN(nn.Module):
	def __init__(self):
		super(AdaIN, self).__init__()
	
	def forward(self, content, style, style_strength=1.0, eps=1e-5):
		b, c, h, w = content.size()
		
		content_std, content_mean = torch.std_mean(content.view(b, c, -1), dim=2, keepdim=True)
		style_std, style_mean = torch.std_mean(style.view(b, c, -1), dim=2, keepdim=True)
	
		normalized_content = (content.view(b, c, -1) - content_mean)/(content_std+eps)
		
		stylized_content = (normalized_content * style_std) + style_mean

		output = (1-style_strength)*content + style_strength*stylized_content.view(b, c, h, w)
		return output


class VGGEncoder(nn.Module):
	def __init__(self, vgg):
		super(VGGEncoder, self).__init__()
		

		self.pad = nn.ReflectionPad2d(1)
		self.relu = nn.ReLU(inplace=True)
		self.pool = nn.AvgPool2d(2)
		self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices = False)
		self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices = False)
		self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices = False)

		###Level0###
		self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
		self.conv0.weight = nn.Parameter(torch.FloatTensor(vgg.modules[0].weight))
		self.conv0.bias = nn.Parameter(torch.FloatTensor(vgg.modules[0].bias))

		###Level1###
		self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
		self.conv1_1.weight = nn.Parameter(torch.FloatTensor(vgg.modules[2].weight))
		self.conv1_1.bias = nn.Parameter(torch.FloatTensor(vgg.modules[2].bias))
		
		self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
		self.conv1_2.weight = nn.Parameter(torch.FloatTensor(vgg.modules[5].weight))
		self.conv1_2.bias = nn.Parameter(torch.FloatTensor(vgg.modules[5].bias))
		
		###Level2###
		self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
		self.conv2_1.weight = nn.Parameter(torch.FloatTensor(vgg.modules[9].weight))
		self.conv2_1.bias = nn.Parameter(torch.FloatTensor(vgg.modules[9].bias))
		
		self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
		self.conv2_2.weight = nn.Parameter(torch.FloatTensor(vgg.modules[12].weight))
		self.conv2_2.bias = nn.Parameter(torch.FloatTensor(vgg.modules[12].bias))
		
		###Level3###
		self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
		self.conv3_1.weight = nn.Parameter(torch.FloatTensor(vgg.modules[16].weight))
		self.conv3_1.bias = nn.Parameter(torch.FloatTensor(vgg.modules[16].bias))

		self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
		self.conv3_2.weight = nn.Parameter(torch.FloatTensor(vgg.modules[19].weight))
		self.conv3_2.bias = nn.Parameter(torch.FloatTensor(vgg.modules[19].bias))

		self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
		self.conv3_3.weight = nn.Parameter(torch.FloatTensor(vgg.modules[22].weight))
		self.conv3_3.bias = nn.Parameter(torch.FloatTensor(vgg.modules[22].bias))

		self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
		self.conv3_4.weight = nn.Parameter(torch.FloatTensor(vgg.modules[25].weight))
		self.conv3_4.bias = nn.Parameter(torch.FloatTensor(vgg.modules[25].bias))

		
		###Level4###
		self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)
		self.conv4_1.weight = nn.Parameter(torch.FloatTensor(vgg.modules[29].weight))
		self.conv4_1.bias = nn.Parameter(torch.FloatTensor(vgg.modules[29].bias))

	def forward(self, x):
		skips = {}
		for level in [1, 2, 3, 4]:
			x = self.encode(x, skips, level)
		return x

	def encode(self, x, skips):
		is_maxpool=False

		out = self.conv0(x)
		out = self.relu(self.conv1_1(self.pad(out)))
		skips['conv1_1'] = out
		
		out = self.relu(self.conv1_2(self.pad(out)))
		skips['conv1_2'] = out
		resize_w, resize_h = out.size(2), out.size(3)
		pooled_feature = self.pool(out)
		
		HH = out - F.interpolate(pooled_feature, size=[resize_w, resize_h], mode='nearest')
		skips['pool1'] = HH
		##################################
		if is_maxpool:
			pooled_feature = self.maxpool1(out)

		out = self.relu(self.conv2_1(self.pad(pooled_feature)))
		skips['conv2_1'] = out

		out = self.relu(self.conv2_2(self.pad(out)))
		skips['conv2_2'] = out
		resize_w, resize_h = out.size(2), out.size(3)
		pooled_feature = self.pool(out)
		
		HH = out - F.interpolate(pooled_feature, size=[resize_w, resize_h], mode='nearest')
		skips['pool2'] = HH
		##################################
		if is_maxpool:
			pooled_feature = self.maxpool2(out)

		out = self.relu(self.conv3_1(self.pad(pooled_feature)))
		skips['conv3_1'] = out

		out = self.relu(self.conv3_2(self.pad(out)))
		out = self.relu(self.conv3_3(self.pad(out)))
		out = self.relu(self.conv3_4(self.pad(out)))
		skips['conv3_4'] = out
		resize_w, resize_h = out.size(2), out.size(3)
		pooled_feature = self.pool(out)
		HH = out - F.interpolate(pooled_feature, size=[resize_w, resize_h], mode='nearest')
		skips['pool3'] = HH

		##################################
		if is_maxpool:
			pooled_feature = self.maxpool3(out)
		
		out = self.relu(self.conv4_1(self.pad(pooled_feature)))
		return out

	def get_features(self, x, level):
		is_maxpool = False

		out = self.conv0(x)
		out = self.relu(self.conv1_1(self.pad(out)))
		if level == 1:
			return out
		
		out = self.relu(self.conv1_2(self.pad(out)))
		pooled_feature = self.pool(out)
		##################################
		if is_maxpool:
			pooled_feature = self.maxpool1(out)
		out = self.relu(self.conv2_1(self.pad(pooled_feature)))
		if level == 2:
			return out

		out = self.relu(self.conv2_2(self.pad(out)))
		pooled_feature = self.pool(out)
		##################################
		if is_maxpool:
			pooled_feature = self.maxpool2(out)
		out = self.relu(self.conv3_1(self.pad(pooled_feature)))
		if level == 3:
			return out

		out = self.relu(self.conv3_2(self.pad(out)))
		out = self.relu(self.conv3_3(self.pad(out)))
		out = self.relu(self.conv3_4(self.pad(out)))
		pooled_feature = self.pool(out)
		##################################
		if is_maxpool:
			pooled_feature = self.maxpool3(out)
		out = self.relu(self.conv4_1(self.pad(pooled_feature)))
		if level == 4:
			return out


class VGGDecoder(nn.Module):
	def __init__(self):
		super(VGGDecoder, self).__init__()
		

		self.pad = nn.ReflectionPad2d(1)
		self.relu = nn.ReLU(inplace=True)
		self.adain = AdaIN()

		self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)
		self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)        
		self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
		self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
		self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)
		self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
		self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)
		self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
		self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 0)


	def forward(self, x, skips):
		for level in [4, 3, 2, 1]:
			x = self.decode(x, skips, level)
		return x

	def decode(self, x, content_skips, style_skips, level, alphas=[], is_recon=True):
		assert level in {4, 3, 2, 1}
		if len(alphas) > 0:
			a1, a2, a3 = alphas[:, 0], alphas[:, 1], alphas[:, 2] #a1는 128~ a3은512
			a1 = a1.unsqueeze(1).unsqueeze(2).unsqueeze(3)
			a2 = a2.unsqueeze(1).unsqueeze(2).unsqueeze(3)
			a3 = a3.unsqueeze(1).unsqueeze(2).unsqueeze(3)
		
		
		out = self.relu(self.conv4_1(self.pad(x)))	
		resize_w, resize_h = content_skips['conv3_4'].size(2), content_skips['conv3_4'].size(3)
		unpooled_feat = F.interpolate(out, size=[resize_w, resize_h], mode='nearest')

		HH = content_skips['pool3']
		
		intermediate_stylized = feature_wct_simple(content_skips['conv3_4'], style_skips['conv3_4'])
		stylized_HH, _ = get_LL_HH(intermediate_stylized)
		unpooled_HH, _ = get_LL_HH(unpooled_feat)
		origin_HH, _ = get_LL_HH(content_skips['conv3_4'])
		
		##############################
		######Stoke Skip Bridge#######
		##############################
		out = unpooled_feat + (gaussian_blur(stylized_HH, (a3)) )

		out = self.relu(self.conv3_4(self.pad(out)))
		out = self.relu(self.conv3_3(self.pad(out)))
		out =  self.relu(self.conv3_2(self.pad(out)))

		out = feature_wct_simple(out, style_skips['conv3_1'])
		
		resize_w, resize_h = content_skips['conv2_2'].size(2), content_skips['conv2_2'].size(3)
		out = self.relu(self.conv3_1(self.pad(out)))
		unpooled_feat = F.interpolate(out, size=[resize_w, resize_h], mode='nearest')
		
		
		HH = content_skips['pool2']
		
		intermediate_stylized = feature_wct_simple(content_skips['conv2_2'], style_skips['conv2_2'])
		stylized_HH, _ = get_LL_HH(intermediate_stylized)
		unpooled_HH, _ = get_LL_HH(unpooled_feat)
		origin_HH, _ = get_LL_HH(content_skips['conv2_2'])
		
		##############################
		######Stoke Skip Bridge#######
		##############################
		out = unpooled_feat + (gaussian_blur(stylized_HH, (a2)) )
		
		out = self.relu(self.conv2_2(self.pad(out)))
		
		out = feature_wct_simple(out, style_skips['conv2_1'])

		resize_w, resize_h = content_skips['conv1_2'].size(2), content_skips['conv1_2'].size(3)
		out = self.relu(self.conv2_1(self.pad(out)))
		unpooled_feat = F.interpolate(out, size=[resize_w, resize_h], mode='nearest')
		
		intermediate_stylized = feature_wct_simple(content_skips['conv1_2'], style_skips['conv1_2'])
		
		HH = content_skips['pool1']
		stylized_HH, _ = get_LL_HH(intermediate_stylized)
		origin_HH, _ = get_LL_HH(content_skips['conv1_2'])
		
		##############################
		######Stoke Skip Bridge#######
		##############################
		out = unpooled_feat + (gaussian_blur(stylized_HH, (a1)) )
	
		out = self.relu(self.conv1_2(self.pad(out)))

		out = feature_wct_simple(out, style_skips['conv1_1'])

		out = self.conv1_1(self.pad(out))
		return out

	def reconstruct(self, x):
		
		out = self.relu(self.conv4_1(self.pad(x)))
		out = F.interpolate(out, size=[out.size(2), out.size(3)], mode='nearest')
		out = self.relu(self.conv3_4(self.pad(out)))
		out = self.relu(self.conv3_3(self.pad(out)))
		out =  self.relu(self.conv3_2(self.pad(out)))
		out = self.relu(self.conv3_1(self.pad(out)))
		out = F.interpolate(out, size=[out.size(2), out.size(3)], mode='nearest')
		out = self.relu(self.conv2_2(self.pad(out)))
		out = self.relu(self.conv2_1(self.pad(out)))
		out = F.interpolate(out, size=[out.size(2), out.size(3)], mode='nearest')
		out = self.relu(self.conv1_2(self.pad(out)))
		out = self.conv1_1(self.pad(out))
		return out


class Baseline_net(nn.Module):
	def __init__(self, pretrained_vgg=None):
		super(Baseline_net, self).__init__()

		
		self.encoder = VGGEncoder(pretrained_vgg).cuda()
		self.decoder = VGGDecoder().cuda()
		self.decorator = StyleDecorator().cuda()
		self.adain = AdaIN()

	def encode(self, x, skips):
		return self.encoder.encode(x, skips)

	def decode(self, x, content_skips, style_skips, level, alphas, is_recon=True):
		return self.decoder.decode(x, content_skips, style_skips, level, alphas, is_recon)

	def forward(self, content, style, content_segment, style_segment, is_recon=True, alpha=1, alphas=[], type='photo'):
		label_set, label_indicator = compute_label_info(content_segment, style_segment)

		wct2_enc_level = [1, 2, 3, 4]
		wct2_dec_level = [1, 2, 3, 4]
		wavelet_skip_level = ['pool1', 'pool2', 'pool3']
		adain_skip_level = ['conv1_1', 'conv2_1', 'conv3_1']
		
		
		content = size_arrange(content)
		style = size_arrange(style)

		####Input####
		content_feat, content_skips = content, {}
		style_feat, style_skips = style, {}
		
		####Encode####
		content_feat = self.encode(content_feat, content_skips)
		style_feat = self.encode(style_feat, style_skips)
		

		####Style transfer####
		transformed_feature = alphas.mean() * self.decorator(content_feat, style_feat) + (1-alphas.mean())*feature_wct_simple(content_feat, style_feat)
		
		####Decode####
		stylized_image = self.decode(transformed_feature, content_skips, style_skips, 1, alphas, is_recon=is_recon)
		
			
		return stylized_image
			

class MultiScaleImageDiscriminator(nn.Module):
	def __init__(self, nc=3, ndf=64):
		super(MultiScaleImageDiscriminator, self).__init__()
		self.nc = nc
		self.output_dim = 1

		self.conv1 = nn.Sequential(
			#256->128
			spectral_norm(nn.Conv2d(self.nc, ndf, 4, 2, 1, bias=False)),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.skip_out1 = nn.Sequential(
			nn.Conv2d(ndf, 32, 3, 1, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			nn.AdaptiveAvgPool2d(8),
			nn.Conv2d(32, self.output_dim, 3, 1, 1, bias=False)
		)
		
		self.conv2 = nn.Sequential(
			#128->64
			spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(ndf*2),
			nn.LeakyReLU(0.2, inplace=True),
		)
		
		self.conv3 = nn.Sequential(
			#64->32
			spectral_norm(nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(ndf*4),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.skip_out2 = nn.Sequential(
			nn.Conv2d(ndf*4, 32, 3, 1, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			nn.AdaptiveAvgPool2d(8),
			nn.Conv2d(32, self.output_dim, 3, 1, 1, bias=False)
		)
		
		self.conv4 = nn.Sequential(
			#32->16
			spectral_norm(nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(ndf*8),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.conv5 = nn.Sequential(
			#16->8
			spectral_norm(nn.Conv2d(ndf*8, self.output_dim, 4, 2, 1, bias=False)),
		)

		init_weights(self)

	def forward(self, input):
		out = self.conv1(input)
		skip_out1 = self.skip_out1(out)
		out = self.conv3(self.conv2(out))
		skip_out2 = self.skip_out2(out)
		out = self.conv5(self.conv4(out))
		out = ((out+skip_out1+skip_out2)*1/3).squeeze()
		#out = ((out+skip_out1+skip_out2)).squeeze()
		
		return out