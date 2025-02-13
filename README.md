# CoMoFusion
Official implementation for “CoMoFusion: Fast and High-quality Fusion of Infrared and Visible Image with Consistency Model.”

## Citation
```
@inproceedings{meng2024comofusion,
  title={Comofusion: fast and high-quality fusion of infrared and visible image with consistency model},
  author={Meng, Zhiming and Li, Hui and Zhang, Zeyang and Shen, Zhongwei and Yu, Yunlong and Song, Xiaoning and Wu, Xiaojun},
  booktitle={Chinese Conference on Pattern Recognition and Computer Vision (PRCV)},
  pages={539--553},
  year={2024},
  organization={Springer}
}
```
## Main Environment

```
conda create -n CoMoFusion python=3.9
conda activate CoMoFusion
pip install -r requirements.txt
```

## Pretrain Weights
We provide the pretrain weights for infrared and visible image fusion. Download the weight and put it into the weights folder.
You can refer to [Baidu Drive](https://pan.baidu.com/s/16z-CQSVMVTFHGWO3NH-N8A) (code:5555) to download it.


## Test Examples
You need to firstly modify the configuration in the ```test_fusion.py``` to put your dataset_root, model_weight and so on.
```
python test_fusion.py
```
## Train Examples
The train code contains two parts: 1. cm_train.py(train consistency model to construct multi-modal joint features).   2. fusion_train.py(train fusion module to generate fused images).





## Abstract
Generative models are widely utilized to model the distribution of fused images in the field of infrared and visible image fusion. However, current generative models based fusion methods often suffer from unstable training and slow inference speed. To tackle this problem, a novel fusion method based on consistency model is proposed, termed as CoMoFusion, which can generate the high-quality images and achieve fast image inference speed. In specific, the consistency model is used to construct multi-modal joint features in the latent space with the forward and reverse process. Then, the infrared and visible features extracted by the trained consistency model are fed into fusion module to generate the final fused image. In order to enhance the texture and salient information of fused images, a novel loss based on pixel value selection is also designed. Extensive experiments on public datasets illustrate that our method obtains the SOTA fusion performance compared with the existing fusion methods.

## OverView
![image](https://github.com/ZhimingMeng/CoMoFusion/blob/main/image/overview.jpg)

## Qualitative fusion results
![image](https://github.com/ZhimingMeng/CoMoFusion/blob/main/image/Qualitative_result_TNO.jpg)
![image](https://github.com/ZhimingMeng/CoMoFusion/blob/main/image/Qualitative_result_MSRS.jpg)


## Quantitative fusion results

![image](https://github.com/ZhimingMeng/CoMoFusion/blob/main/image/Quantitative_TNO.jpg)
![image](https://github.com/ZhimingMeng/CoMoFusion/blob/main/image/Quantitative_MSRS.jpg)
