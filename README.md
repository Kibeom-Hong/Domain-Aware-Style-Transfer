# Domain Aware Universal Style Transfer

### Official Pytorch Implementation of 'Domain Aware Universal Style Transfer' (ICCV 2021)

![teaser](https://user-images.githubusercontent.com/77425614/127080253-dcee74fd-0301-4767-8f9c-6958d2da1ff8.PNG)

> ## Domain Aware Universal Style Transfer
> 
> Kibeom Hong (Yonsei Univ.), Seogkyu Jeon (Yonsei Univ.), Jianlong Fu (Microsoft Research), Huan Yang (Microsoft Research), Hyeran Byun (Yonsei Univ.)
>
> Paper : https://arxiv.org/abs/2108.04441
> 
> **Abstract**: Style transfer aims to reproduce content images with the styles from reference images. Existing universal style transfer methods successfully deliver arbitrary styles to original images either in an artistic or a photo-realistic way. However, the range of “arbitrary style” defined by existing works is bounded in the particular domain due to their structural limitation. Specifically, the degrees of content preservation and stylization are established according to a predefined target domain. As a result, both photo-realistic and artistic models have difficulty in performing the desired style transfer for the other domain. To overcome this limitation, we propose a unified architecture, **D**omain-aware **S**tyle **T**ransfer **N**etworks (**DSTN**) that transfer not only the style but also the property of domain (i.e., domainness) from a given reference image. To this end, we design a novel domainness indicator that captures the domainness value from the texture and structural features of reference images. Moreover, we introduce a unified framework with domain-aware skip connection to adaptively transfer the stroke and palette to the input contents guided by the domainness indicator. Our extensive experiments validate that our model produces better qualitative results and outperforms previous methods in terms of proxy metrics on both artistic and photo-realistic stylizations.


## Prerequisites

### Dependency
- Python 3.6
- CUDA 11.0
- Pytorch 1.7
- Check the requirements.txt

```
pip install -r requirements.txt
```

## Usage
#### Set pretrained weights
* Pretrained models for encoder(VGG-19) can be found in the `./baseline_checkpoints`
- Prepare pretrained models for **Domainnes Indicator**
  -  Domainnes Indicator can be downloaded at [style_indicator.pth](https://drive.google.com/file/d/1-rf2CdrCr9ei9KS-V0H3kjo1oaPmT5Xz/view?usp=sharing)
- Prepare pretrained models for **Decoder**
  -  Vanilla version can be downloaded at [Decoder.pth](https://drive.google.com/file/d/1tlUTBHB_rg9eRDa-wi1xPkbtBHGs1CUQ/view?usp=sharing)
  -  Adversarial version can be downloaded at [Decoder_adversarial.pth](https://drive.google.com/file/d/1lMCtPR-ZZUqJ1MHExXoTmCTO3K34rCCz/view?usp=sharing)

- Move these pretrained weights to each folders:
  - style_indicator.pth -> `./train_results/StyleIndicator/log/`
  - decoder.pth -> `./train_results/Decoder/log/`
  - decoder_adversarial.pth -> `./train_results/Decoder_adversarial/log/` 
 
  **(Please rename decoder_adversarial.pth -> decoder.pth)**

#### Inference (Automatic)
- Vanilla decoder
```
bash scripts/transfer.sh
```

- Decoder with adversarial loss
```
bash scripts/transfer_adversarial.sh
```

#### Inference (User Guided)
- Vanilla decoder (You should set --alpha **value** in script file)
```
bash scripts/transfer_user_guided.sh
```

- Decoder with adversarial loss (You should set --alpha **value** in script file)
```
bash scripts/transfer_adversarial_user_guided.sh
```

#### Inference (Interpolation)
```
bash scripts/interpolate.sh
```


#### Training
Our networks could be trained with end-to-end manner. However, we recommend to train **StyleIndicator** and **Decoder** respectively.

- (1 step) Train StyleIndicator 
```
bash scripts/train_indicator.sh
```

- (2 step) Train Decoder
```
bash scripts/train_decoder.sh
```


#### Evaluation
Available soon


## Ciation
If you find this work useful for your research, please cite:
```
@article{Hong2021DomainAwareUS,
  title={Domain-Aware Universal Style Transfer},
  author={Kibeom Hong and Seogkyu Jeon and Huan Yang and Jianlong Fu and H. Byun},
  journal={ArXiv},
  year={2021},
  volume={abs/2108.04441}
}
```

## Contact
If you have any question or comment, please contact the first author of this paper - Kibeom Hong

[cha2068@yonsei.ac.kr](cha2068@yonsei.ac.kr)
