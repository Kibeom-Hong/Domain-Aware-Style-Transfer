import os
import torch
import torchfile
import numpy as np

from apis.utils.utils import *
from data.dataset_util import *
from abc import abstractmethod, ABCMeta
from apis.utils.baseline_models import Baseline_net
from apis.utils.style_indicator import New_DA_Net_v1 as DA_Net


class Inference(metaclass=ABCMeta):
    """
    Base factory for Style Transfer Inferences.
    """

    def __init__(self, baseline_config: dict):

        self.imsize = baseline_config["imsize"]
        self.cencrop = baseline_config["cencrop"]
        self.cropsize = baseline_config["cropsize"]
        self.num_workers = baseline_config["num_workers"]
        self.train_result_dir = baseline_config["train_result_dir"]

        pretrained_vgg = torchfile.load(
            baseline_config["pretrained_vgg_checkpoint_path"])
        self.decoder = Baseline_net(pretrained_vgg=pretrained_vgg)
        self.decoder.cuda()
        self.decoder.load_state_dict(torch.load(
            f"{baseline_config['decoder_path']}decoder.pth")['state_dict'])

        self.decoder_adversarial = Baseline_net(pretrained_vgg=pretrained_vgg)
        self.decoder_adversarial.cuda()
        self.decoder_adversarial.load_state_dict(torch.load(
            f"{baseline_config['decoder_adversarial_path']}decoder.pth")['state_dict'])

        self.DA_Net = DA_Net(self.imsize)
        self.DA_Net.cuda()
        self.DA_Net.load_state_dict(torch.load(
            f"{baseline_config['style_indicator_path']}style_indicator.pth")['state_dict'])

    @abstractmethod
    def main(self, content_frame: np.ndarray, style_frame: np.ndarray, config: dict):
        pass

    @abstractmethod
    def preprocess(self, content_frame: np.ndarray, style_frame: np.ndarray):
        content_image = Transfer_TestDataset(
            content_frame, (256, 512), self.cropsize, self.cencrop, is_test=True)

        style_reference_image = Transfer_TestDataset(
            style_frame, (256, 512), self.cropsize, self.cencrop, is_test=True)
        return content_image, style_reference_image

    @abstractmethod
    def load_data(self, content_image: np.ndarray, style_reference_image: np.ndarray):
        content_loader = torch.utils.data.DataLoader(
            content_image, batch_size=self.config["batch_size"], shuffle=False, drop_last=True, num_workers=self.num_workers)

        style_reference_loader = torch.utils.data.DataLoader(
            style_reference_image, batch_size=self.config["batch_size"], shuffle=False, drop_last=True, num_workers=self.num_workers)
        return content_loader, style_reference_loader

    @abstractmethod
    def load_models(self):
        self.DA_Net.train(False)
        self.DA_Net.eval()

        if self.config["ST_comment"] == "Decoder":
            self.network = self.decoder
            self.network.train(False)
            self.network.eval()
        else:
            self.network = self.decoder_adversarial
            self.network.train(False)
            self.network.eval()

    @abstractmethod
    def inference(self):
        pass

    @abstractmethod
    def postprocess(self):
        pass

    @abstractmethod
    def clean_memory(self):
        pass
