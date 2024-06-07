"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import matplotlib.pyplot as plt
import argparse
import os
from cm import utils
import numpy as np
import torch as th
import torch.distributed as dist
from torch.utils.data import DataLoader
import cv2
from cm import dist_util, logger
from cm.image_datasets import ImageDataset, _list_image_files_recursively
from cm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample
import torch.nn.functional as F
from core import  metrics as Metrics




@th.no_grad()
def main():

    def vision_features(feature_maps, img_type):
        feature_path ='outputs2/feature_maps/noise_0/forward'
        os.makedirs(feature_path, exist_ok=True)
        count = 0
        for features in feature_maps:
            count += 1
            B, C, H ,W =features.shape
            file_name = 'feature_maps_' + img_type + '_level_' + str(count) + '_avg_channel_' + '.jpg'
            output_path = os.path.join(feature_path,file_name)
            map = th.mean(features,dim=1)
            grid_img = map.detach()
            grid_img = Metrics.tensor2img(grid_img)
            # grid_img = np.resize(grid_img, (128, 128))
            # print(grid_img.shape)
            heatmap = cv2.applyColorMap(np.uint8(grid_img), cv2.COLORMAP_JET)

            # Metrics.save_img(grid_img, '{}'.format(output_path))
            cv2.imwrite( output_path,  heatmap)




    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    if "consistency" in args.training_mode:
        distillation = True
    else:
        distillation = False

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        distillation=distillation,
    )
    model.load_state_dict(
        th.load(args.model_path)
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None

    all_images = []
    all_labels = []
    generator = get_generator(args.generator, args.num_samples, args.seed)

    all_files = _list_image_files_recursively(args.data_dir)
    all_iv_files = _list_image_files_recursively(args.image_iv_paths)
    dataset = ImageDataset(
        data_dir=args.data_dir,
        data_ir_dir=args.image_iv_paths,
        resolution=190,
        image_paths=all_files,
        image_iv_paths=all_iv_files,
        random_crop=False,
        random_flip=False,
        isCrop=False
    )
    testloader = DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=True
    )

    noise_step = 28
    test_sample_path = '../smapler_pictures/Kaist_noise_12'
    os.makedirs(test_sample_path, exist_ok=True)
    # test_sample_path_vis = '../smapler_pictures/TNO_model_out_10/vis'
    # os.makedirs(test_sample_path_vis, exist_ok=True)
    # test_sample_path_ir='../smapler_pictures/TNO_model_out_10/ir'
    # os.makedirs(test_sample_path_ir, exist_ok=True)

    # test_sampel_result_path = '../MSRS_noise_pictures/noise_20/vis'
    # test_sampel_result_path1 = '../MSRS_noise_pictures/noise_20/ir'
    # os.makedirs(test_sampel_result_path, exist_ok=True)
    # os.makedirs(test_sampel_result_path1, exist_ok=True)
    model_kwargs = {}

    for i, (batch, path) in enumerate(testloader):

        print(i+1)
        batch['img'] =  batch['img'].to(dist_util.dev())
        batch['vis'] = batch['vis'].to(dist_util.dev())
        batch['ir'] = batch['ir'].to(dist_util.dev())

        # 补齐操作
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
            left_2_pad, right_2_pad, left_1_pad, right_1_pad, 0, 0, 0, 0), "constant", 0.), \
                                                  F.pad(batch['vis'], (
                                                      left_2_pad, right_2_pad, left_1_pad, right_1_pad, 0, 0, 0,
                                                      0), "constant", 0.), F.pad(batch['ir'], (
            left_2_pad, right_2_pad, left_1_pad, right_1_pad, 0, 0, 0, 0))



        xT, model_out ,denoised = karras_sample(
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
                add_noise_t =noise_step,
                remove_noise_t=noise_step,
                isfeat=False
            )
        # feat = karras_sample(
        #     diffusion,
        #     model,
        #     # (args.batch_size, 3, args.image_size, args.image_size),
        #     (args.batch_size, 3, 64, 64),
        #     steps=args.steps,
        #     model_kwargs=model_kwargs,
        #     device=dist_util.dev(),
        #     clip_denoised=args.clip_denoised,
        #     sampler=args.sampler,
        #     sigma_min=args.sigma_min,
        #     sigma_max=args.sigma_max,
        #     s_churn=args.s_churn,
        #     s_tmin=args.s_tmin,
        #     s_tmax=args.s_tmax,
        #     s_noise=args.s_noise,
        #     generator=generator,
        #     ts=ts,
        #     x_start=batch['img'],
        #     add_noise_t=noise_step,
        #     remove_noise_t=noise_step,
        #     isfeat=True
        # )
        # feature_maps = feat[1]
        # for i in feature_maps:
        #     print(i.shape)
        # vision_features(feat,path[0])
        # if(i == 10) : break
        grid_img = th.cat((batch['vis'].repeat(1, 3, 1, 1).detach(),
                           xT[:,0,:,:].repeat(1, 3, 1, 1).detach(),
                           model_out[:,0,:,:].repeat(1, 3, 1, 1).detach(),
                           denoised[:, 0, :, :].repeat(1, 3, 1, 1).detach(),
                           batch['ir'].repeat(1, 3, 1, 1).detach(),
                           xT[:, 1, :, :].repeat(1, 3, 1, 1).detach(),
                           model_out[:, 1, :, :].repeat(1, 3, 1, 1).detach(),
                           denoised[:, 1, :, :].repeat(1, 3, 1, 1).detach(),
                           ), dim=0)
        # grid_img = model_out[:,0,:,:].repeat(1, 3, 1, 1).detach().detach()
        grid_img = Metrics.tensor2img(grid_img)
        Metrics.save_img(grid_img, '{}/{}'.format(test_sample_path, path[0]))
        #
        # grid_img = model_out[:,1,:,:].repeat(1, 3, 1, 1).detach().detach()
        # grid_img = Metrics.tensor2img(grid_img)
        # Metrics.save_img(grid_img, '{}/{}'.format(test_sample_path_ir, path[0]))
         # grid_img =xT[:, 3, :, :].repeat(1, 3, 1, 1).detach()
        # grid_img = Metrics.tensor2img(grid_img)
        # Metrics.save_img(grid_img, '{}/{}'.format(test_sampel_result_path1, path[0]))
    #     sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    #     sample = sample.permute(0, 2, 3, 1)
    #     sample = sample.contiguous()
    #
    #     gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
    #     dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
    #     all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
    #     if args.class_cond:
    #         gathered_labels = [
    #             th.zeros_like(classes) for _ in range(dist.get_world_size())
    #         ]
    #         dist.all_gather(gathered_labels, classes)
    #         all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
    #     logger.log(f"created {len(all_images) * args.batch_size} samples")
    #
    # arr = np.concatenate(all_images, axis=0)
    # arr = arr[: args.num_samples]
    # if args.class_cond:
    #     label_arr = np.concatenate(all_labels, axis=0)
    #     label_arr = label_arr[: args.num_samples]
    # if dist.get_rank() == 0:
    #     shape_str = "x".join([str(x) for x in arr.shape])
    #     out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    #     logger.log(f"saving to {out_path}")
    #     if args.class_cond:
    #         np.savez(out_path, arr, label_arr)
    #     else:
    #         np.savez(out_path, arr)
    #
    # dist.barrier()
    # logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        training_mode="consistency_training",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="onestep",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="../model_weights/cmt1.pt",
        seed=42,
        ts="",
        data_dir='../datasets/kaist_validation/vis',
        image_iv_paths='../datasets/kaist_validation/ir',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
