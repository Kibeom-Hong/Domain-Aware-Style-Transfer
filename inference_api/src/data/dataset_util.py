import torch
import matplotlib
import torchvision
import numpy as np
import torchvision.transforms as transforms

from PIL import Image

matplotlib.use('Agg')
Image.MAX_IMAGE_PIXELS = 1000000000


class Transfer_TestDataset(torch.utils.data.Dataset):
    def __init__(self, frame: np.ndarray, imsize=None, cropsize=None, cencrop=False, is_test=False):
        super(Transfer_TestDataset, self).__init__()

        if is_test:
            self.transform = _transformer()
        else:
            self.transform = _transformer(imsize, cropsize, cencrop)

        self.frame = frame

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.transform(self.frame)


def _normalizer(denormalize=False):
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


def tensor_to_PIL(tensor, nrow=4, npadding=0):
    denormalize = _normalizer(denormalize=True)
    tensor = torchvision.utils.make_grid(tensor, nrow=nrow, padding=npadding)
    return torchvision.transforms.ToPILImage()(denormalize(tensor).clamp_(0.0, 1.0))
