import gc
import torch
import numpy as np

from data.dataset_util import *
from ..baseline import Inference
from singleton_decorator import singleton


@singleton
class StyleTransfer(Inference):

    def __init__(self, baseline_config: dict):
        super().__init__(baseline_config)

    def main(self, content_frame: np.ndarray, style_frame: np.ndarray, config: dict) -> Image:

        self.config = config

        content_image, style_reference_image = self.preprocess(
            content_frame, style_frame)

        content_loader, style_reference_loader = self.load_data(
            content_image, style_reference_image)

        self.load_models()
        prediction = self.inference(
            content_loader, style_reference_loader)
        result = self.postprocess(prediction)
        self.clean_memory(prediction)
        return result

    def preprocess(self, content_frame: np.ndarray, style_frame: np.ndarray):
        return super().preprocess(content_frame, style_frame)

    def load_data(self, content_image: np.ndarray, style_reference_image: np.ndarray):
        return super().load_data(content_image, style_reference_image)

    def load_models(self):
        return super().load_models()

    def inference(self, content_loader, style_reference_loader):
        content_iter = iter(content_loader)
        style_iter = iter(style_reference_loader)

        with torch.no_grad():
            empty_segment = np.asarray([])
            content = next(content_iter).cuda()
            style_reference = next(style_iter).cuda()
            style_alphas = self.get_alphas(style_reference)

            prediction = self.network(content, style_reference, empty_segment, empty_segment,
                                      is_recon=True, alphas=style_alphas, type='photo')

        self.clean_memory(content)
        self.clean_memory(style_reference)
        self.clean_memory(style_alphas)

        return prediction

    def clean_memory(self, tensor: torch.Tensor):
        del tensor
        gc.collect()
        torch.cuda.empty_cache()

    def postprocess(self, prediction: torch.Tensor) -> Image:
        return tensor_to_PIL(prediction, nrow=self.config["batch_size"])

    def get_alphas(self, imgs):
        self.DA_Net.eval()
        with torch.no_grad():
            alphas = []
            for level in [1, 2, 3]:
                feat = self.network.encoder.get_features(imgs, level)
                alphas.append(torch.sigmoid(self.DA_Net(feat, level)))
            alphas = torch.stack(alphas).unsqueeze(0).cuda()
        return alphas
