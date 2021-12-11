"""
from photo_wct.py of https://github.com/NVIDIA/FastPhotoStyle
Copyright (C) 2018 NVIDIA Corporation.
Licensed under the CC BY-NC-SA 4.0
"""
import os
import torch
import imageio
import datetime
import numpy as np
import torch.nn as nn

from PIL import Image
from torchvision import transforms


def svd(feat, iden=False, device='cuda:0'):
    size = feat.size()
    mean = torch.mean(feat, 1)
    mean = mean.unsqueeze(1).expand_as(feat)
    _feat = feat.clone()
    _feat -= mean
    if size[1] > 1:
        conv = torch.mm(_feat, _feat.t()).div(size[1] - 1)
    else:
        conv = torch.mm(_feat, _feat.t())
    if iden:
        conv += torch.eye(size[0]).to(device)

    u, e, v = torch.svd(conv, some=False)
    return u, e, v


def get_squeeze_feat(feat):
    _feat = feat.squeeze(0)
    size = _feat.size(0)
    return _feat.view(size, -1).clone()


def get_rank(singular_values, dim, eps=0.00001):
    r = dim
    for i in range(dim - 1, -1, -1):
        if singular_values[i] >= eps:
            r = i + 1
            break
    return r


def covsqrt_mean(feature, inverse=False, tolerance=1e-14):

    b, c, h, w = feature.size()

    mean = torch.mean(feature.view(b, c, -1), dim=2, keepdim=True)
    zeromean = feature.view(b, c, -1) - mean
    cov = torch.bmm(zeromean, zeromean.transpose(1, 2))

    eps_matrix_ = (torch.ones_like(cov)*1e-8).cuda()
    evals, evects = torch.symeig(cov+eps_matrix_, eigenvectors=True)

    p = 0.5
    if inverse:
        p *= -1

    covsqrt = []
    for i in range(b):
        k = 0
        for j in range(c):
            if evals[i][j] > tolerance:
                k = j
                break
        covsqrt.append(torch.mm(evects[i][:, k:],
                                torch.mm(evals[i][k:].pow(p).diag_embed(),
                                         evects[i][:, k:].t())).unsqueeze(0))
    covsqrt = torch.cat(covsqrt, dim=0)

    return covsqrt, mean


def whitening(feature):
    b, c, h, w = feature.size()

    inv_covsqrt, mean = covsqrt_mean(feature, inverse=True)

    normalized_feature = torch.matmul(inv_covsqrt, feature.view(b, c, -1)-mean)

    return normalized_feature.view(b, c, h, w)


def whitening_edit(feature):
    b, c, h, w = feature.size()
    cont_feat = get_squeeze_feat(feature)
    cont_min = cont_feat.min()
    cont_max = cont_feat.max()
    cont_mean = torch.mean(cont_feat, 1).unsqueeze(1).expand_as(cont_feat)
    cont_feat -= cont_mean

    _, c_e, c_v = svd(cont_feat, iden=True)
    k_c = get_rank(c_e, cont_feat.size()[0])
    c_d = (c_e[0:k_c]).pow(-0.5)
    step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
    step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
    normalized_feature = torch.mm(step2, cont_feat)

    return normalized_feature.view(b, c, h, w)


def coloring(feature, target):
    b, c, h, w = feature.size()

    covsqrt, mean = covsqrt_mean(target)

    colored_feature = torch.matmul(covsqrt, feature.view(b, c, -1)) + mean

    return colored_feature.view(b, c, h, w)


def SwitchWhiten2d(x):
    N, C, H, W = x.size()

    in_data = x.view(N, C, -1)

    eye = in_data.data.new().resize_(C, C)
    eye = torch.nn.init.eye_(eye).view(1, C, C).expand(N, C, C)

    mean_in = in_data.mean(-1, keepdim=True)
    x_in = in_data - mean_in
    cov_in = torch.bmm(x_in, torch.transpose(x_in, 1, 2)).div(H * W)

    mean = mean_in
    cov = cov_in + 1e-5 * eye

    Ng, c, _ = cov.size()
    P = torch.eye(c).to(cov).expand(Ng, c, c)

    rTr = (cov * P).sum((1, 2), keepdim=True).reciprocal_()
    cov_N = cov * rTr
    for k in range(5):
        P = torch.baddbmm(1.5, P, -0.5, torch.matrix_power(P, 3), cov_N)

    wm = P.mul_(rTr.sqrt())
    x_hat = torch.bmm(wm, in_data-mean)

    return x_hat, wm, mean


def wct_core(cont_feat, styl_feat, weight=1, registers=None, device='cuda:0'):
    cont_feat = get_squeeze_feat(cont_feat)
    cont_min = cont_feat.min()
    cont_max = cont_feat.max()
    cont_mean = torch.mean(cont_feat, 1).unsqueeze(1).expand_as(cont_feat)
    cont_feat -= cont_mean

    if not registers:
        _, c_e, c_v = svd(cont_feat, iden=True, device=device)

        styl_feat = get_squeeze_feat(styl_feat)
        s_mean = torch.mean(styl_feat, 1)
        _, s_e, s_v = svd(styl_feat, iden=True, device=device)
        k_s = get_rank(s_e, styl_feat.size()[0])
        s_d = (s_e[0:k_s]).pow(0.5)
        EDE = torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(
            s_d) * weight), (s_v[:, 0:k_s].t()))

        if registers is not None:
            registers['EDE'] = EDE
            registers['s_mean'] = s_mean
            registers['c_v'] = c_v
            registers['c_e'] = c_e
    else:
        EDE = registers['EDE']
        s_mean = registers['s_mean']
        _, c_e, c_v = svd(cont_feat, iden=True, device=device)

    k_c = get_rank(c_e, cont_feat.size()[0])
    c_d = (c_e[0:k_c]).pow(-0.5)
    # TODO could be more fast
    step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
    step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
    whiten_cF = torch.mm(step2, cont_feat)

    targetFeature = torch.mm(EDE, whiten_cF)
    targetFeature = targetFeature + \
        s_mean.unsqueeze(1).expand_as(targetFeature)
    targetFeature.clamp_(cont_min, cont_max)

    return targetFeature


def Bw_wct_core(content_feat, style_feat, weight=1, registers=None, device='cpu'):
    N, C, H, W = content_feat.size()
    cont_min = content_feat.min().item()
    cont_max = content_feat.max().item()

    whiten_cF, _,  _ = SwitchWhiten2d(content_feat)
    _, wm_s, s_mean = SwitchWhiten2d(style_feat)

    targetFeature = torch.bmm(torch.inverse(wm_s), whiten_cF)
    targetFeature = targetFeature.view(N, C, H, W)
    targetFeature = targetFeature + \
        s_mean.unsqueeze(2).expand_as(targetFeature)
    targetFeature.clamp_(cont_min, cont_max)

    return targetFeature


def wct_core_segment(content_feat, style_feat, content_segment, style_segment,
                     label_set, label_indicator, weight=1, registers=None,
                     device='cpu'):
    def resize(feat, target):
        size = (target.size(2), target.size(1))
        if len(feat.shape) == 2:
            return np.asarray(Image.fromarray(feat).resize(size, Image.NEAREST))
        else:
            return np.asarray(Image.fromarray(feat, mode='RGB').resize(size, Image.NEAREST))

    def get_index(feat, label):
        mask = np.where(feat.reshape(feat.shape[0] * feat.shape[1]) == label)
        if mask[0].size <= 0:
            return None
        return torch.LongTensor(mask[0]).cuda()

    squeeze_content_feat = content_feat.squeeze(0)
    squeeze_style_feat = style_feat.squeeze(0)

    content_feat_view = squeeze_content_feat.view(
        squeeze_content_feat.size(0), -1).clone()
    style_feat_view = squeeze_style_feat.view(
        squeeze_style_feat.size(0), -1).clone()

    resized_content_segment = resize(content_segment, squeeze_content_feat)
    resized_style_segment = resize(style_segment, squeeze_style_feat)

    target_feature = content_feat_view.clone()
    for label in label_set:
        if not label_indicator[label]:
            continue
        content_index = get_index(resized_content_segment, label)
        style_index = get_index(resized_style_segment, label)
        if content_index is None or style_index is None:
            continue
        masked_content_feat = torch.index_select(
            content_feat_view, 1, content_index)
        masked_style_feat = torch.index_select(style_feat_view, 1, style_index)
        _target_feature = wct_core(
            masked_content_feat, masked_style_feat, weight, registers, device=device)
        if torch.__version__ >= '0.4.0':
            # XXX reported bug in the original repository
            new_target_feature = torch.transpose(target_feature, 1, 0)
            new_target_feature.index_copy_(0, content_index,
                                           torch.transpose(_target_feature, 1, 0))
            target_feature = torch.transpose(new_target_feature, 1, 0)
        else:
            target_feature.index_copy_(1, content_index, _target_feature)
    return target_feature


def feature_wct(content_feat, style_feat, content_segment=None, style_segment=None,
                label_set=None, label_indicator=None, weight=1, registers=None, alpha=1, device='cuda:0'):
    if label_set is not None:
        target_feature = wct_core_segment(content_feat, style_feat, content_segment, style_segment,
                                          label_set, label_indicator, weight, registers, device=device)
    else:
        target_feature = Bw_wct_core(content_feat, style_feat, device=device)

    target_feature = target_feature.view_as(content_feat)
    target_feature = alpha * target_feature + (1 - alpha) * content_feat
    return target_feature


def feature_wct_simple(content_feat, style_feat, alpha=1):
    target_feature = Bw_wct_core(content_feat, style_feat)

    target_feature = target_feature.view_as(content_feat)
    target_feature = alpha * target_feature + (1 - alpha) * content_feat
    return target_feature


"""
from photo_wct.py of https://github.com/NVIDIA/FastPhotoStyle
Copyright (C) 2018 NVIDIA Corporation.
Licensed under the CC BY-NC-SA 4.0
"""


def init_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()


class Timer:
    def __init__(self, msg='Elapsed time: {}', verbose=True):
        self.msg = msg
        self.start_time = None
        self.verbose = verbose

    def __enter__(self):
        self.start_time = datetime.datetime.now()

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.verbose:
            print(self.msg.format(datetime.datetime.now() - self.start_time))


def _normalizer(denormalize=False):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    if denormalize:
        MEAN = [-mean/std for mean, std in zip(MEAN, STD)]
        STD = [1/std for std in STD]

    return transforms.Normalize(mean=MEAN, std=STD)


def open_image(image_path, image_size=None):
    normalize = _normalizer()
    image = Image.open(image_path)
    _transforms = []
    if image_size is not None:
        image = transforms.Resize(image_size)(image)
        # _transforms.append(transforms.Resize(image_size))
    w, h = image.size
    _transforms.append(transforms.CenterCrop((h // 16 * 16, w // 16 * 16)))
    _transforms.append(transforms.ToTensor())
    _transforms.append(normalize)
    transform = transforms.Compose(_transforms)
    return transform(image).unsqueeze(0)


def change_seg(seg):
    color_dict = {
        (0, 0, 255): 3,  # blue
        (0, 255, 0): 2,  # green
        (0, 0, 0): 0,  # black
        (255, 255, 255): 1,  # white
        (255, 0, 0): 4,  # red
        (255, 255, 0): 5,  # yellow
        (128, 128, 128): 6,  # grey
        (0, 255, 255): 7,  # lightblue
        (255, 0, 255): 8  # purple
    }
    arr_seg = np.asarray(seg)
    new_seg = np.zeros(arr_seg.shape[:-1])
    for x in range(arr_seg.shape[0]):
        for y in range(arr_seg.shape[1]):
            if tuple(arr_seg[x, y, :]) in color_dict:
                new_seg[x, y] = color_dict[tuple(arr_seg[x, y, :])]
            else:
                min_dist_index = 0
                min_dist = 99999
                for key in color_dict:
                    dist = np.sum(np.abs(np.asarray(key) - arr_seg[x, y, :]))
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_index = color_dict[key]
                    elif dist == min_dist:
                        try:
                            min_dist_index = new_seg[x, y-1, :]
                        except Exception:
                            pass
                new_seg[x, y] = min_dist_index
    return new_seg.astype(np.uint8)


def load_segment(image_path, image_size=None):
    if not image_path:
        return np.asarray([])
    image = Image.open(image_path)
    if image_size is not None:
        transform = transforms.Resize(image_size, interpolation=Image.NEAREST)
        image = transform(image)
    w, h = image.size
    transform = transforms.CenterCrop((h // 16 * 16, w // 16 * 16))
    image = transform(image)
    if len(np.asarray(image).shape) == 3:
        image = change_seg(image)
    return np.asarray(image)


def compute_label_info(content_segment, style_segment):
    if not content_segment.size or not style_segment.size:
        return None, None
    max_label = np.max(content_segment) + 1
    label_set = np.unique(content_segment)
    label_indicator = np.zeros(max_label)
    for l in label_set:
        content_mask = np.where(content_segment.reshape(
            content_segment.shape[0] * content_segment.shape[1]) == l)
        style_mask = np.where(style_segment.reshape(
            style_segment.shape[0] * style_segment.shape[1]) == l)

        c_size = content_mask[0].size
        s_size = style_mask[0].size
        if c_size > 10 and s_size > 10 and c_size / s_size < 100 and s_size / c_size < 100:
            label_indicator[l] = True
        else:
            label_indicator[l] = False
    return label_set, label_indicator


def mkdir(dname):
    if not os.path.exists(dname):
        os.makedirs(dname)
    else:
        assert os.path.isdir(dname), 'alread exists filename {}'.format(dname)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def TVloss(img, tv_weight):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
    h_variance = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    loss = tv_weight * (h_variance + w_variance)
    return loss


def zeros_like(x):
    return torch.autograd.Variable(torch.zeros_like(x).cuda())


def ones_like(x):
    return torch.autograd.Variable(torch.ones_like(x).cuda())


def denorm(x):
    out = (x+1)/2
    return out.clamp_(0, 1)


def save_video(video, save_path, type='photo'):
    video = denorm(video)
    '''
	vid_lst=[]
	for i in range(0, num_samples*num_samples, num_samples):
		temp_vid = list(video[i:i+num_samples])
		temp_vid = torch.cat(temp_vid, dim=-1)
		vid_lst.append(temp_vid)
		
	save_videos = torch.cat(vid_lst, dim=2)
	'''

    save_videos = video.data.cpu().numpy().transpose(0, 2, 3, 1)
    outputdata = save_videos * 255
    outputdata = outputdata.astype(np.uint8)
    dir_path = save_path
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if type == 'photo':
        gif_file_path = os.path.join(dir_path, 'Photo_StylizedVideo.gif')
    elif type == 'art':
        gif_file_path = os.path.join(dir_path, 'Art_StylizedVideo.gif')
    else:
        gif_file_path = os.path.join(dir_path, 'content_StylizedVideo.gif')
    imageio.mimsave(gif_file_path, outputdata, fps=25)
