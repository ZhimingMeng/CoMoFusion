# CoMoFusion
Official implementation for “CoMoFusion: Fast and High-quality Fusion of Infrared and Visible Image with Consistency Model.”

## Citation
```
@misc{meng2024comofusion,
      title={CoMoFusion: Fast and High-quality Fusion of Infrared and Visible Image with Consistency Model}, 
      author={Zhiming Meng and Hui Li and Zeyang Zhang and Zhongwei Shen and Yunlong Yu and Xiaoning Song and Xiaojun Wu},
      year={2024},
      eprint={2405.20764},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Abstract
Generative models are widely utilized to model the distribution of fused images in the field of infrared and visible image fusion. However, current generative models based fusion methods often suffer from unstable training and slow inference speed. To tackle this problem, a novel fusion method based on consistency model is proposed, termed as CoMoFusion, which can generate the high-quality images and achieve fast image inference speed. In specific, the consistency model is used to construct multi-modal joint features in the latent space with the forward and reverse process. Then, the infrared and visible features extracted by the trained consistency model are fed into fusion module to generate the final fused image. In order to enhance the texture and salient information of fused images, a novel loss based on pixel value selection is also designed. Extensive experiments on public datasets illustrate that our method obtains the SOTA fusion performance compared with the existing fusion methods.

## OverView


