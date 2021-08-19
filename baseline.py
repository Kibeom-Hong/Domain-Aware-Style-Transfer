import os, math, random, pdb, time, timeit
from tqdm import tqdm
import torch, torchfile
import torchvision.utils as utils
from torch.utils.data import random_split
from torch.utils.data.sampler import SubsetRandomSampler
from utils import *
from data.dataset_util import *

from baseline_models import Baseline_net, gaussian_blur, MultiScaleImageDiscriminator
from style_indicator import New_DA_Net_v1 as DA_Net
from torch.distributions.beta import Beta
from contextual_utils import contextual_loss, contextual_loss_v2
from AdaIN import Adaptive_IN

from evaluation import *

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

def get_HH_LL(x):
	pooled = torch.nn.functional.avg_pool2d(x, 2)
	up_pooled = torch.nn.functional.interpolate(pooled, scale_factor=2, mode='nearest')
	HH = x - up_pooled
	LL = up_pooled
	return HH, LL

def get_domainess(cur_iter, total_iter, batch):
    alpha = np.exp((cur_iter - (0.5 * total_iter)) / (0.25 * total_iter))
    distribution = Beta(alpha, 1)
    #distribution = Beta(0.5, 0.5)
    return distribution.sample((batch, 1))

class Baseline(object):
	def __init__(self, args):
		super(Baseline, self).__init__()
		self.imsize = args.imsize #(512,1024)
		self.batch_size = args.batch_size
		self.cencrop = args.cencrop
		self.cropsize = args.cropsize
		self.num_workers = args.num_workers
		self.content_dir = args.content_dir
		self.style_dir = args.style_dir
		self.lr = args.lr
		self.train_result_dir = args.train_result_dir
		self.DA_comment = args.DA_comment
		self.ST_comment = args.ST_comment
		self.max_iter = args.max_iter
		self.check_iter = args.check_iter
		self.args = args

		self.is_da_train = args.is_da_train
		self.is_st_train = args.is_st_train


		#######################################
		####          Model Load           ####
		#######################################
		pretrained_vgg = torchfile.load('./baseline_checkpoints/vgg_normalised_conv4_1.t7')
		self.network = Baseline_net(pretrained_vgg=pretrained_vgg)
		self.network.cuda()

		self.DA_Net = DA_Net(self.imsize)
		self.DA_Net.cuda()

		self.MSD_img = MultiScaleImageDiscriminator(nc=3, ndf=64)
		self.MSD_img.cuda()
		

		#######################################
		####   Loss function, Optimizer    ####
		#######################################
		for param in self.network.encoder.parameters():
			param.requires_grad = False
		
		betas=(0.5, 0.999)
		self.dec_optim = torch.optim.Adam(
			filter(lambda p: p.requires_grad, self.network.decoder.parameters()),
			lr = self.lr,
			betas=betas
			)

		self.Di_optim = torch.optim.Adam(self.MSD_img.parameters(), lr=self.lr, betas=betas, weight_decay=0.00001)
		self.enc_optim = torch.optim.Adam(self.DA_Net.parameters(), lr=self.lr, betas=betas)

		self.MSE_loss = torch.nn.MSELoss().cuda()
		self.bce_loss = torch.nn.BCEWithLogitsLoss().cuda()

		self.tv_weight = 1#1e-6

		self.result_img_dir = os.path.join(self.train_result_dir, self.ST_comment, 'imgs')
		self.result_log_dir = os.path.join(self.train_result_dir, self.DA_comment, 'log')
		self.result_st_dir = os.path.join(self.train_result_dir, self.ST_comment, 'log')
		os.makedirs(self.result_img_dir, exist_ok=True)
		os.makedirs(self.result_log_dir, exist_ok=True)
		os.makedirs(self.result_st_dir, exist_ok=True)

	def train(self):

		# ########################
		# wandb.init(project="style_transfer")
		# if self.is_da_train:
		# 	wandb.run.name = self.DA_comment
		# else:
		# 	wandb.run.name = self.ST_comment
		# wandb.config.update(self.args)
		# ########################


		#######################################
		#### Data Loader (Photo, Artistic) ####
		#######################################
		self.data_set = MSCOCO(self.content_dir, self.imsize, self.cropsize, self.cencrop)
		self.art_data_set = WiKiART(self.style_dir, self.imsize, self.cropsize, self.cencrop)

		self.data_loader = torch.utils.data.DataLoader(self.data_set, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
		self.art_data_loader = torch.utils.data.DataLoader(self.art_data_set, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)


		#######################################
		####         Train Phase           ####
		#######################################
		self.network.encoder.train(False)
		self.network.decoder.train(True)
		self.DA_Net.train(True)
		

		content_weight = 1.0
		style_weight = 0.0

		d_lambda = 1
		c_lambda = 5
		_domainess = np.linspace(0, 1, self.max_iter)
		best_distance=0.0
		print(self.DA_comment," model train start!")

		self.content_data_loader_iter = iter(self.data_loader)
		self.art_data_loader_iter = iter(self.art_data_loader)

		for iteration in range(self.max_iter):
			
			try:
				content = next(self.content_data_loader_iter).cuda()
				art_ref = next(self.art_data_loader_iter).cuda()#.to(self.device)
			except:
				self.content_data_loader_iter = iter(self.data_loader)
				self.art_data_loader_iter = iter(self.art_data_loader)
				content = next(self.content_data_loader_iter).cuda()
				art_ref = next(self.art_data_loader_iter).cuda()#.to(self.device)

			if self.is_da_train:
				self.adain = Adaptive_IN()
				
				with torch.no_grad():
					domainess = get_domainess(iteration, (self.max_iter), 1).cuda()
					mix_img = (domainess)*(art_ref) + (1-domainess)*(content)
					#mix_img = self.adain.interpolate(content, art_ref, domainess)
				
				content_bce_loss = 0
				style_bce_loss = 0
				cls_loss = 0
				domain_loss = 0
				
				dist = []
				c_dist = []
				s_dist = []

				
				#######################
				####origin train code##
				#######################
				for level in [1,2,3]:
					cont_feat = self.network.encoder.get_features(content, level)
					cont_feat = self.DA_Net(cont_feat, level)

					art_feat = self.network.encoder.get_features(art_ref, level)
					art_feat = self.DA_Net(art_feat, level)
					
					mix_feat = self.network.encoder.get_features(mix_img, level)
					mix_feat = self.DA_Net(mix_feat, level)

					content_bce_loss = self.bce_loss(cont_feat, torch.zeros_like(cont_feat).cuda())
					style_bce_loss = self.bce_loss(art_feat, torch.ones_like(art_feat).cuda())
					cls_loss += (content_bce_loss + style_bce_loss)
					
					
					domain_loss += (1-domainess)*(torch.mean(torch.abs((cont_feat) - (mix_feat) ))) + \
						(domainess)*(torch.mean(torch.abs((art_feat) - (mix_feat))))
					

					############################
					###  StyleIndicator Acc  ###
					############################
					with torch.no_grad():
						dist.append(float(torch.mean(torch.abs(torch.sigmoid(cont_feat) - torch.sigmoid(art_feat)))))
						c_dist.append(float(torch.mean(torch.abs(torch.sigmoid(cont_feat) - torch.sigmoid(mix_feat)))))
						s_dist.append(float(torch.mean(torch.abs(torch.sigmoid(art_feat) - torch.sigmoid(mix_feat)))))


				total_loss = d_lambda * domain_loss + c_lambda * cls_loss
				
				self.enc_optim.zero_grad()
				total_loss.backward()
				self.enc_optim.step()

				################################
				####      Checkpoints       ####
				################################
				
				if (iteration) % self.check_iter == 0:
					print("%s: Iteration: [%d/%d]\tC_loss: %2.4f\tD_loss: %2.4f \tDomainess: %2.4f"%(time.ctime(), iteration, self.max_iter, (cls_loss).item(), domain_loss.item(), domainess))
					print("level:%d  dist:%2.4f   c_dist:%2.4f  s_dist:%2.4f"%(0, dist[0], c_dist[0], s_dist[0]))
					print("level:%d  dist:%2.4f   c_dist:%2.4f  s_dist:%2.4f"%(1, dist[1], c_dist[1], s_dist[1]))
					print("level:%d  dist:%2.4f   c_dist:%2.4f  s_dist:%2.4f"%(2, dist[2], c_dist[2], s_dist[2]))
					print("AVG  :   dist:%2.4f   c_dist:%2.4f  s_dist:%2.4f"%(np.mean(dist), np.mean(c_dist), np.mean(s_dist)))
				
				# wandb.log({
				# 	#"L/content_bce_loss" : content_bce_loss.item(),
				# 	#"L/style_bce_loss" : style_bce_loss.item(),
				# 	"L/cls_loss" : cls_loss.item(),
				# 	"L/domain_loss" : domain_loss.item(),
				# 	"D/Average_distance" : np.mean(dist)
				# 	})

				del content_bce_loss, style_bce_loss, domain_loss, domainess

				if ( np.mean(dist) >= 0.95 and np.mean(dist)>best_distance ) or (np.mean(dist) == 1.0):
					best_distance = np.mean(dist)
					torch.save({'iteration': iteration,
						'state_dict': self.DA_Net.state_dict(),},
						os.path.join(self.result_log_dir, 'model_'+str(iteration)+'.pth'))


			if self.is_st_train:
				################################
				####Load pretrained_DA_model ###
				####    Get Alpha value     ####
				################################
				self.DA_Net.load_state_dict(torch.load(os.path.join(self.result_log_dir, 'style_indicator.pth'))['state_dict'])
				
				###############################
				####Step 2 : Reconstruction ###
				###############################
				empty_segment = np.asarray([])
				
				cont_alphas = self.get_alphas(content)
				content_recon = self.network(content, content, empty_segment, empty_segment, is_recon=True, alphas=cont_alphas, type='photo')

				art_alphas = self.get_alphas(art_ref)
				art_recon = self.network(art_ref, art_ref, empty_segment, empty_segment, is_recon=True, alphas=art_alphas, type='photo')
				
				
				#########################	
				####Adversarial parts####
				#########################
				origin_gan_output = self.MSD_img(content)
				recon_gan_output = self.MSD_img(content_recon.detach())
				D_content_loss = self.bce_loss(origin_gan_output, ones_like(origin_gan_output)) + self.bce_loss(recon_gan_output, zeros_like(recon_gan_output))

				origin_ref_gan_output = self.MSD_img(art_ref)
				recon_ref_gan_output = self.MSD_img(art_recon.detach())
				D_ref_loss = self.bce_loss(origin_ref_gan_output, ones_like(origin_ref_gan_output)) + self.bce_loss(recon_ref_gan_output, zeros_like(recon_ref_gan_output))

				D_loss = D_content_loss + D_ref_loss

				self.Di_optim.zero_grad()
				D_loss.backward()
				self.Di_optim.step()
				

				tv_loss = TVloss(content_recon, self.tv_weight)
						
				feature_recon_loss = []
				feature_recon_loss_art = []
				cx_loss_photo = 0
				cx_loss_art = 0

				for level in [1,2,3,4]:
					cont_feat = self.network.encoder.get_features(content, level)
					cont_recon_feat = self.network.encoder.get_features(content_recon, level)
					feature_recon_loss.append(self.MSE_loss(cont_feat, cont_recon_feat))
					if level == 4:
						cx_loss_photo += (contextual_loss_v2(cont_feat, cont_recon_feat))
					
					del cont_feat, cont_recon_feat
					torch.cuda.empty_cache()

				for level in [1,2,3,4]:
					art_feat = self.network.encoder.get_features(art_ref, level)
					art_recon_feat = self.network.encoder.get_features(art_recon, level)
					feature_recon_loss_art.append(self.MSE_loss(art_feat, art_recon_feat))
					if level == 4:
						cx_loss_art += (contextual_loss_v2(art_feat, art_recon_feat))

		
				content_recon_gan_output = self.MSD_img(content_recon)
				art_recon_gan_output = self.MSD_img(art_recon)
				
				G_loss = self.bce_loss(content_recon_gan_output, ones_like(content_recon_gan_output)) + self.bce_loss(art_recon_gan_output, ones_like(art_recon_gan_output))
				total_loss = torch.mean(torch.stack(feature_recon_loss))*0.1 + torch.mean(torch.stack(feature_recon_loss_art))*0.1 + 1*(cx_loss_photo+cx_loss_art) + tv_loss + 0.1*G_loss
				
				
				if torch.isnan(total_loss):
					continue

				self.dec_optim.zero_grad()
				total_loss.backward()
				self.dec_optim.step()

				
				################################
				####      Checkpoints       ####
				################################
				wandb.log({
					"L/feature_recon_loss" : torch.mean(torch.stack(feature_recon_loss)).item(),
					"L/feature_recon_loss_art" : torch.mean(torch.stack(feature_recon_loss_art)).item(),
					"L/total_loss" : total_loss.item(),
					"L/D_loss" : D_loss.item(),
					"L/G_loss" : G_loss.item(),
					"L/cx_loss_photo" : cx_loss_photo.item(),
					"L/cx_loss_art" : cx_loss_art.item(),
					})

				if (iteration) % self.check_iter == 0:
					print("%s: Iteration: [%d/%d]\tC_loss: %2.4f"%(time.ctime(), iteration, self.max_iter, total_loss.item()))
					print("Alphas : ", cont_alphas.mean(dim=1).cpu().data)
					#wandb.log({"Recon_results": wandb.Image(denorm(torch.cat([content, content_recon, art_ref, art_recon]), nrow=self.batch_size))})
					#wandb.log({"Fixed_results": wandb.Image(denorm(self.fixed_test(self.result_img_dir, iteration), nrow=8))})

				if (iteration) % 2500 == 0:
					torch.save({'iteration': iteration,
						'state_dict': self.network.state_dict(),},
						os.path.join(self.result_st_dir, 'model_'+str(iteration)+'.pth'))
				

	def transfer(self, args):
		self.DA_Net.load_state_dict(torch.load(os.path.join(self.result_log_dir, 'style_indicator.pth'))['state_dict'])
		self.network.load_state_dict(torch.load(os.path.join(self.result_st_dir, 'decoder.pth'))['state_dict'])
		
		
		content_set = Transfer_TestDataset(args.test_content, (256,512), self.cropsize, self.cencrop, type='art', is_test=True)
		art_reference_set = Transfer_TestDataset(args.test_a_reference, (256,512), self.cropsize, self.cencrop, type='art', is_test=True)
		photo_reference_set = Transfer_TestDataset(args.test_p_reference, (256,512), self.cropsize, self.cencrop, type='art', is_test=True)
		
		content_loader = torch.utils.data.DataLoader(content_set, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers)
		art_reference_loader = torch.utils.data.DataLoader(art_reference_set, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers)
		photo_reference_loader = torch.utils.data.DataLoader(photo_reference_set, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers)


		self.DA_Net.train(False)
		self.DA_Net.eval()
		self.network.train(False)
		self.network.eval()

		dir_path = os.path.join(self.result_img_dir, 'transfer', self.DA_comment+'_'+self.ST_comment)
		if not os.path.exists(dir_path):
			os.makedirs(dir_path)

		N = art_reference_set.__len__()
		content_iter = iter(content_loader)
		art_iter = iter(art_reference_loader)
		photo_iter = iter(photo_reference_loader)
		for iteration in range(N//self.batch_size):
			with torch.no_grad():
				empty_segment = np.asarray([])
				content = next(content_iter).cuda()
				a_reference = next(art_iter).cuda()
				p_reference = next(photo_iter).cuda()

				art_alphas = self.get_alphas(a_reference)
				photo_alphas = self.get_alphas(p_reference)
				art_stylized_output = self.network(content, a_reference, empty_segment, empty_segment, is_recon=True, alphas=art_alphas, type='photo')
				print(str(iteration), 'art : ' , art_alphas)
				photo_stylized_output = self.network(content, p_reference, empty_segment, empty_segment, is_recon=True, alphas=photo_alphas, type='photo')
				print(str(iteration), 'photo : ' , photo_alphas)
				
			imsave(art_stylized_output,  os.path.join(dir_path,  'single_art_stylized_'+str(iteration)+'.png'), nrow=self.batch_size )
			imsave(photo_stylized_output,  os.path.join(dir_path, 'single_photo_stylized_'+str(iteration)+'.png'), nrow=self.batch_size )
			
			del content, a_reference, p_reference, art_stylized_output, photo_stylized_output
			torch.cuda.empty_cache()
			time.sleep(0.2)

	def transfer_user_guided(self, args):
		self.DA_Net.load_state_dict(torch.load(os.path.join(self.result_log_dir, 'style_indicator.pth'))['state_dict'])
		self.network.load_state_dict(torch.load(os.path.join(self.result_st_dir, 'decoder.pth'))['state_dict'])
		
		
		content_set = Transfer_TestDataset(args.test_content, (256,512), self.cropsize, self.cencrop, type='art', is_test=True)
		art_reference_set = Transfer_TestDataset(args.test_a_reference, (256,512), self.cropsize, self.cencrop, type='art', is_test=True)
		photo_reference_set = Transfer_TestDataset(args.test_p_reference, (256,512), self.cropsize, self.cencrop, type='art', is_test=True)
		
		content_loader = torch.utils.data.DataLoader(content_set, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers)
		art_reference_loader = torch.utils.data.DataLoader(art_reference_set, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers)
		photo_reference_loader = torch.utils.data.DataLoader(photo_reference_set, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers)


		self.DA_Net.train(False)
		self.DA_Net.eval()
		self.network.train(False)
		self.network.eval()

		dir_path = os.path.join(self.result_img_dir, 'transfer_user_guided', self.DA_comment+'_'+self.ST_comment)
		if not os.path.exists(dir_path):
			os.makedirs(dir_path)

		N = art_reference_set.__len__()
		content_iter = iter(content_loader)
		art_iter = iter(art_reference_loader)
		photo_iter = iter(photo_reference_loader)
		for iteration in range(N//self.batch_size):
			with torch.no_grad():
				empty_segment = np.asarray([])
				content = next(content_iter).cuda()
				a_reference = next(art_iter).cuda()
				p_reference = next(photo_iter).cuda()

				alphas = torch.ones(args.batch_size, 1).repeat(1, 3)*args.alpha
				art_stylized_output = self.network(content, a_reference, empty_segment, empty_segment, is_recon=True, alphas=alphas, type='photo')
				photo_stylized_output = self.network(content, p_reference, empty_segment, empty_segment, is_recon=True, alphas=alphas, type='photo')
				
				
			imsave(art_stylized_output,  os.path.join(dir_path,  'single_art_stylized_'+str(iteration)+'.png'), nrow=self.batch_size )
			imsave(photo_stylized_output,  os.path.join(dir_path, 'single_photo_stylized_'+str(iteration)+'.png'), nrow=self.batch_size )
			
			del content, a_reference, p_reference, art_stylized_output, photo_stylized_output
			torch.cuda.empty_cache()
			time.sleep(0.2)


	def transfer_seg(self, args):
		from baseline_models_seg import Baseline_net as Baseline_net_seg
		pretrained_vgg = torchfile.load('./baseline_checkpoints/vgg_normalised_conv4_1.t7')
		self.network = Baseline_net_seg(pretrained_vgg=pretrained_vgg)
		self.network.cuda()


		self.DA_Net.load_state_dict(torch.load(os.path.join(self.result_log_dir, 'style_indicator.pth'))['state_dict'])
		self.network.load_state_dict(torch.load(os.path.join(self.result_st_dir, 'decoder.pth'))['state_dict'])
		
		self.DA_Net.train(False)
		self.DA_Net.eval()
		self.network.train(False)
		self.network.eval()

		dir_path = os.path.join(self.result_img_dir, 'transfer_seg2',  self.DA_comment+'_'+self.ST_comment)
		if not os.path.exists(dir_path):
			os.makedirs(dir_path)
		for fname in tqdm.tqdm(range(0,60)):
			content_fname = 'single_content_'+str(fname)+'.png'
			style_fname = 'single_p_reference_'+str(fname)+'.png'
			
			content_segment_fname = 'in'+str(fname)+'.png'
			style_segment_fname = 'tar'+str(fname)+'.png'
			output_fname = str(fname)+'.png'
			with torch.no_grad():
				_content = os.path.join(args.test_content, content_fname)
				_style = os.path.join(args.test_p_reference, style_fname)

				_content_segment = os.path.join(args.test_content_segment, content_segment_fname) if args.test_content_segment else None
				_style_segment = os.path.join(args.test_p_reference_segment, style_segment_fname) if args.test_p_reference_segment else None
				_output = os.path.join(dir_path, output_fname)

				content = open_image(_content, 512).cuda()
				style = open_image(_style, 512).cuda()
				content_segment = load_segment(_content_segment, 512)
				style_segment = load_segment(_style_segment, 512)

				alphas = self.get_alphas(style)
				
				stylized_output = self.network(content, style, content_segment, style_segment, is_recon=True, alphas=alphas, type='photo')
				
				print(str(fname), alphas)
				imsave(stylized_output,  os.path.join(dir_path,  'single_photo_stylized_'+str(fname)+'.png'), nrow=self.batch_size )
				#save_image(stylized_output,  os.path.join(dir_path,  'single_photo_stylized_'+str(fname)+'.png'))
				
				del content, style, stylized_output
				torch.cuda.empty_cache()
				time.sleep(0.2)


	def get_alphas(self, imgs):
		self.DA_Net.eval()
		with torch.no_grad():
			alphas = []
			for level in [1,2,3]: #2,3,4
				feat = self.network.encoder.get_features(imgs, level)
				alphas.append(torch.sigmoid(self.DA_Net(feat, level)))
			alphas = torch.stack(alphas).unsqueeze(0).cuda()
			

		return alphas
