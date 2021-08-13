from __future__ import print_function
import argparse, os


def str2bool(v):
	if v.lower() in ('true', 't'):
		return True
	elif v.lower() in ('false', 'f'):
		return False


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Domain Aware Universal Style Transfer')
	
	parser.add_argument('--output_image_path', default='./results')
	parser.add_argument('--content_dir', type=str, default='../../dataset/MSCoCo', help='Content data path to train the network')
	parser.add_argument('--style_dir', type=str, default='../../dataset/wikiart', help='Content data path to train the network')

	######For train arguments#####
	parser.add_argument('--train_type', type=str, default='split')
	parser.add_argument('--type', type=str, default='train')
	parser.add_argument('--max_iter', type=int, default=1000)
	parser.add_argument('--batch_size', type=int, default=4)
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--check_iter', type=int, default=100)
	parser.add_argument('--imsize', type=int, default=128)
	parser.add_argument('--cropsize', type=int, default=128)
	parser.add_argument('--num_workers', type=int, default=12)
	parser.add_argument('--cencrop', action='store_true', default=False)
	parser.add_argument('--train_result_dir', type=str, default='./train_results', help='Content data path to train the network')
	parser.add_argument('--is_da_train', type=str2bool, default='False')
	parser.add_argument('--is_st_train', type=str2bool, default='True')

	######For test arguments#####
	parser.add_argument('--check_point', type=str)
	parser.add_argument('--DA_comment', type=str)
	parser.add_argument('--ST_comment', type=str)
	parser.add_argument('--model_type', type=str)
	parser.add_argument('--test_content', type=str, default='./test_images/content/')
	parser.add_argument('--test_a_reference', type=str, default='./test_images/a_reference/')
	parser.add_argument('--test_p_reference', type=str, default='./test_images/p_reference/')
	parser.add_argument('--test_content_segment', type=str, default='./test_images/content/')
	parser.add_argument('--test_p_reference_segment', type=str, default='./test_images/p_reference/')

	parser.add_argument('--DA_Net_trained_epoch', type=int, default='84420')
	parser.add_argument('--decoder_trained_epoch', type=int, default='50000')
	
	args = parser.parse_args()

	if args.train_type == 'split':
		from baseline import Baseline as Baseline
	elif args.train_type == 'total':
		from baseline_total import Baseline_total as Baseline
	elif args.train_type == 'seg':
		from baseline_seg import Baseline_seg as Baseline
	model = Baseline(args)



	if args.type == 'train':
		model.train()
	elif args.type == 'transfer':
		model.transfer(args)
	elif args.type == 'transfer_iterative':
		model.transfer_iterative(args)
	elif args.type == 'eval':
		model.eval(args)
	elif args.type == 'interpolate':
		model.interpolate(args)
		
		


