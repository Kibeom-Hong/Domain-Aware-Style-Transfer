import torch
import torch.nn.functional as F
import pdb
from utils import whitening, coloring, whitening_edit

def extract_patches(feature, patch_size, stride, padding='zero'):
	kh, kw = patch_size
	dh, dw = stride
	
	# padding input
	ph = int((kh-1)/2)
	pw = int((kw-1)/2)
	padding_size = (pw, pw, ph, ph)
	
	if padding == 'zero':
		feature = F.pad(feature, padding_size, 'constant', 0)
	elif padding == 'reflect':
		feature = F.pad(feature, padding_size, mode= padding)
	else:
		raise RuntimeError("padding mode error")

	# get all image windows of size (kh, kw) and stride (dh, dw)
	kernels = feature.unfold(2, kh, dh).unfold(3, kw, dw)
	
	# view the windows as (kh * kw)
	kernels = kernels.contiguous().view(*kernels.size()[:-2], -1)
	
	return kernels

class StyleDecorator(torch.nn.Module):
	
	def __init__(self):
		super(StyleDecorator, self).__init__()

	def kernel_normalize(self,kernel, k=3,  eps=1e-5):
		b, ch, h, w, kk = kernel.size()
		
		# calc kernel norm
		kernel = kernel.view(b, ch, h*w, kk).transpose(2,1)
		kernel_norm = torch.norm(kernel.contiguous().view(b, h*w, ch*kk), p=2, dim=2, keepdim=True) + eps
		
		# kernel reshape
		kernel = kernel.view(b, h*w, ch, k, k)
		kernel_norm = kernel_norm.view(b, h*w, 1, 1, 1)
		
		return kernel, kernel_norm

	def conv2d_with_style_kernels(self, features, kernels, patch_size, padding='zero', deconv_flag=False):
		output = list()
		b, c, h, w = features.size()
		
		# padding
		pad_size = (patch_size-1)//2
		padding_size = (pad_size, pad_size, pad_size, pad_size)
		
		# batch-wise convolutions with style kernels
		for feature, kernel in zip(features, kernels):
			feature = feature.unsqueeze(0)
			if padding == 'zero':
				feature = F.pad(feature, padding_size, 'constant', 0)
			elif padding == 'reflect':
				feature = F.pad(feature, padding_size, mode=padding)
			elif padding == 'none':
				pass
			else:
				raise RuntimeError("padding mode error")
				
			# deconvolution by transpose conv weight's in_ch, out_ch
			if deconv_flag:
				output.append(F.conv_transpose2d(feature, kernel, padding=int(patch_size-1)))
				# output.append(F.conv_transpose2d(feature, kernel)[:,:,pad_size:-pad_size, pad_size:-pad_size])
			else:
				output.append(F.conv2d(feature, kernel))
		
		return torch.cat(output, dim=0)
		
	def binarization_patch_score(self, features):
		outputs= list()
		
		# batch-wise binarization
		for feature in features:
			# best matching patch index
			matching_indices = torch.argmax(feature, dim=0)
			one_hot_mask = torch.zeros_like(feature)
			#pdb.set_trace()
			h, w = matching_indices.size()
			
			for i in range(h):
				for j in range(w):
					ind = matching_indices[i,j]
					one_hot_mask[ind, i, j] = 1
			outputs.append(one_hot_mask.unsqueeze(0))
			
		return torch.cat(outputs, dim=0)

	def topk_binarization_patch_score(self, features, alpha):
		outputs= list()
		
		# batch-wise binarization
		for iter, feature in enumerate(features):
			# best matching patch index
			#matching_indices = torch.argmax(feature, dim=0)
			matching_indices = torch.topk(feature, int(1024*alpha[iter].item()), dim=0)[1]
			one_hot_mask = torch.zeros_like(feature)
			
			#pdb.set_trace()
			c, h, w = matching_indices.size()
			
			for i in range(h):
				for j in range(w):
					ind = matching_indices[i,j]
					one_hot_mask[ind, i, j] = 1
			outputs.append(one_hot_mask.unsqueeze(0))
			
		return torch.cat(outputs, dim=0)

   
	# deconvolution normalize weight mask
	def norm_deconvolution(self, h, w, patch_size):
		mask = torch.ones((h, w))
		fullmask = torch.zeros( (h+patch_size-1, w+patch_size-1) )
		for x in range(patch_size):
			for y in range(patch_size):
				paddings = (x, patch_size-x-1, y, patch_size-y-1)
				padded_mask = F.pad(mask, paddings, 'constant', 0)
				fullmask += padded_mask
		pad_width = int((patch_size-1)/2)
		if pad_width == 0:
			deconv_norm = fullmask
		else:
			deconv_norm = fullmask[pad_width:-pad_width, pad_width:-pad_width]
		return deconv_norm.view(1, 1, h, w)


	def reassemble_feature(self, normalized_content_feature, normalized_style_feature, patch_size, patch_stride):
		## get patches of style feature
		style_kernel = extract_patches(normalized_style_feature, [patch_size, patch_size],
										  [patch_stride, patch_stride])
		## kernel normalize
		style_kernel, kernel_norm = self.kernel_normalize(style_kernel, patch_size)
		
		## convolution with style kernel(patch wise convolution)
		patch_score = self.conv2d_with_style_kernels(normalized_content_feature, style_kernel/kernel_norm, patch_size)
		
		## binarization
		binarized = self.binarization_patch_score(patch_score)

		#alphas = (torch.ones(3, 1)*0.1).cuda()
		#topk_binarized = self.topk_binarization_patch_score(patch_score, alphas)
		#pdb.set_trace()
		## deconv norm
		deconv_norm = self.norm_deconvolution(h=binarized.size(2), w=binarized.size(3), patch_size= patch_size)

		## deconvolution
		output = self.conv2d_with_style_kernels(binarized, style_kernel, patch_size, deconv_flag=True)
		
		return output/deconv_norm.type_as(output)

   
		
	def forward(self, content_feature, style_feature, style_strength=1.0, patch_size=3, patch_stride=1): 

		# 1-1. content feature projection
		normalized_content_feature = whitening(content_feature)

		# 1-2. style feature projection
		normalized_style_feature = whitening(style_feature)

		# 2. swap content and style features
		reassembled_feature = self.reassemble_feature(normalized_content_feature, normalized_style_feature,
				patch_size=patch_size, patch_stride=patch_stride)

		# 3. reconstruction feature with style mean and covariance matrix
		stylized_feature = coloring(reassembled_feature, style_feature)

		# 4. content and style interpolation
		result_feature = (1-style_strength) * content_feature + style_strength * stylized_feature
		
		return result_feature