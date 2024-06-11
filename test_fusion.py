"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import time

import statistics
import torch
import numpy as np

import torch.nn.functional as F
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
from torch.utils.data import DataLoader

from core import  metrics as Metrics
import argparse
import os

import numpy as np
import torch
import torch as th
import torch.distributed as dist
from PIL import Image

from cm import dist_util, logger
from cm.fs_head import Fusion_Head, Fusion_Head_backfs

from cm.image_datasets import load_data, ImageDataset, _list_image_files_recursively
from cm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample

@th.no_grad
def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    if "consistency" in args.training_mode:
        distillation = True
    else:
        distillation = False

    logger.log("creating model and diffusion...")

    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        distillation=distillation,
    )

    #add_noise  40 mean min noise adding to image   0 mean max noise adding to image
    noise_step = 40

    #load data
    all_files = _list_image_files_recursively(args.data_dir)
    all_iv_files = _list_image_files_recursively(args.image_iv_paths)
    dataset = ImageDataset(
        data_dir=args.data_dir,
        data_ir_dir=args.image_iv_paths,
        resolution = 256,
        image_paths =all_files,
        image_iv_paths =all_iv_files ,
        random_crop=False,
        random_flip=False,
        isCrop=False,
        istrain=False

    )
    testloader = DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=True
    )

    #load consistency model
    model.load_state_dict(
        th.load(args.model_path)
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # load fusion model
    fusion_model = Fusion_Head().to(dist_util.dev())
    fusion_model.load_state_dict(
    th.load(args.fusion_model_path)['fusion_model'])
    fusion_model.eval()


    generator = get_generator(args.generator, args.num_samples, args.seed)
    os.makedirs(args.test_result_path, exist_ok=True)
    model_kwargs = {}
    fuse_time = []
    logger.log("sampling...")
    for i, (batch, path) in enumerate(testloader):

        fd = []
        print(i)
        batch['img'] = batch['img'].to(dist_util.dev())
        batch['vis'] = batch['vis'].to(dist_util.dev())
        batch['ir'] = batch['ir'].to(dist_util.dev())

        #padding
        B, C, H, W = batch['vis'].shape
        x = H
        rest = 4 - x % 4 if x % 4 else 0
        left = rest // 2
        left_1_pad, right_1_pad, left_1, right_1 = left, rest - left, left, left + x

        x = W
        rest = 4 - x % 4 if x % 4 else 0
        left = rest // 2
        left_2_pad, right_2_pad, left_2, right_2 = left, rest - left, left, left + x

        batch['img'], batch['vis'], batch['ir'] = F.pad(batch['img'], (
        left_2_pad, right_2_pad, left_1_pad, right_1_pad, 0, 0, 0, 0), "constant", 1.), \
                                                              F.pad(batch['vis'], (
                                                              left_2_pad, right_2_pad, left_1_pad, right_1_pad, 0, 0, 0,
                                                              0), "constant", 1.), F.pad(batch['ir'], (
        left_2_pad, right_2_pad, left_1_pad, right_1_pad, 0, 0, 0, 0))


        start = time.time()
        #extract features from consistency model
        with th.no_grad():
                feat = karras_sample(
                diffusion,
                model,
                # (args.batch_size, 3, args.image_size, args.image_size),
                (args.batch_size, 3, 64, 64),
                steps=args.steps,
                model_kwargs=model_kwargs,
                device=dist_util.dev(),
                clip_denoised=args.clip_denoised,
                sampler=args.sampler,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                s_churn=args.s_churn,
                s_tmin=args.s_tmin,
                s_tmax=args.s_tmax,
                s_noise=args.s_noise,
                generator=generator,
                ts=ts,
                x_start=batch['img'],
                add_noise_t=noise_step,
                remove_noise_t=noise_step,
                isfeat=True
                 )
                fd.append(feat)
                del feat

        # feed features into fusion model to fuse image
        fusion = fusion_model(fd[0])
        end = time.time()
        fuse_time.append(end - start)
        del fd
        fusion = fusion[:, :, left_1: right_1, left_2: right_2]

        #save fused result
        grid_img = fusion.detach()
        grid_img = Metrics.tensor2img(grid_img)
        Metrics.save_img(grid_img, '{}/{}'.format(args.test_result_path, path[0]))
    mean = statistics.mean(fuse_time[1:])

    print(f'fuse avg time: {mean:.4f}')



    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        training_mode="consistency_training",
        generator="determ",
        clip_denoised=False,
        num_samples=5,
        batch_size=1,
        sampler="onestep",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="./model_weights/consistency_model.pt",
        fusion_model_path="./model_weights/fusion_model.pth",
        test_result_path ='./fusion_imgs/TNO',
        seed=42,
        ts="",
        # data_dir='../datasets/MSRS/test/vi',
        # image_iv_paths='../datasets/MSRS/test/ir',
        # data_dir='../datasets/RoadScene/Vis',
        # image_iv_paths='../datasets/RoadScene/Inf',
        data_dir='./datasets/TNO/vi',
        image_iv_paths='./datasets/TNO/ir',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
