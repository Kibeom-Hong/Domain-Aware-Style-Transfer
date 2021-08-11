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

## Usuage
#### Inference (Automatic)
#### Inference (User guided)
#### Training


## Ciation
If you find this work useful for your research, please cite:
```
citation
```

## Contact
If you have any question or comment, please contacth the first author of this paper - Kibeom Hong

[cha2068@yonsei.ac.kr](cha2068@yonsei.ac.kr)
