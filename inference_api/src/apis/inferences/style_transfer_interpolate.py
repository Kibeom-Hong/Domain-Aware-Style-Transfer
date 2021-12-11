import gc
import torch
import numpy as np

from data.dataset_util import *
from ..baseline import Inference
from singleton_decorator import singleton


@singleton
class StyleTransferInterpolate(Inference):

    def __init__(self, baseline_config: dict):
        super().__init__(baseline_config)

    def main(self, content_frame: np.ndarray, style_frame: np.ndarray, config: dict) -> Image:

        self.config = config

        content_image, style_reference_image = self.preprocess(
            content_frame, style_frame)

        content_loader, style_reference_loader = self.load_data(
            content_image, style_reference_image)

        self.load_models()
        results = self.inference(content_loader, style_reference_loader)
        return self.postprocess(results)

    def preprocess(self, content_frame: np.ndarray, style_frame: np.ndarray):
        return super().preprocess(content_frame, style_frame)

    def load_data(self, content_image: np.ndarray, style_reference_image: np.ndarray):
        return super().load_data(content_image, style_reference_image)

    def load_models(self):
        return super().load_models()

    def inference(self, content_loader, style_reference_loader):
        empty_segment = np.asarray([])
        content_iter = iter(content_loader)
        art_iter = iter(style_reference_loader)

        with torch.no_grad():
            empty_segment = np.asarray([])
            content = next(content_iter).cuda()
            style_reference = next(art_iter).cuda()

            predictions = []
            for _, v in enumerate(np.linspace(0, 1, 11)):
                alphas = torch.ones(
                    self.config["batch_size"], 1).repeat(1, 3)*v
                style_result = self.network(
                    content, style_reference, empty_segment, empty_segment, is_recon=True, alphas=alphas, type='photo')
                predictions.append(style_result)
                self.clean_memory(style_result)

        self.clean_memory(content)
        self.clean_memory(style_reference)

        return torch.cat(predictions).cpu()

    def postprocess(self, style_results: torch.Tensor) -> Image:
        return tensor_to_PIL(style_results, nrow=self.config["batch_size"])

    def clean_memory(self, tensor: torch.Tensor):
        del tensor
        gc.collect()
        torch.cuda.empty_cache()

    def get_HH_LL(self, x):
        pooled = torch.nn.functional.avg_pool2d(x, 2)
        up_pooled = torch.nn.functional.interpolate(
            pooled, scale_factor=2, mode='nearest')
        HH = x - up_pooled
        LL = up_pooled
        return HH, LL
