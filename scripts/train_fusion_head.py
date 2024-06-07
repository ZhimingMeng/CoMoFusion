"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import random
import torch
import numpy as np
from torch.nn import init
from tqdm import tqdm






import argparse
import os

from torch.utils.tensorboard import SummaryWriter

from cm.fs_head import Fusion_Head, Fusion_Head_backfs
import numpy as np
import torch as th
import torch.distributed as dist
from torch.utils.data import DataLoader

from cm import dist_util, logger
from cm.fs_loss import  Fusion_loss
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

from core import  metrics as Metrics


def warm_up_cosine_lr_scheduler(optimizer, epochs=100, warm_up_epochs=5, eta_min=1e-9):
    """
    Description:
        - Warm up cosin learning rate scheduler, first epoch lr is too small

    Arguments:
        - optimizer: input optimizer for the training
        - epochs: int, total epochs for your training, default is 100. NOTE: you should pass correct epochs for your training
        - warm_up_epochs: int, default is 5, which mean the lr will be warm up for 5 epochs. if warm_up_epochs=0, means no need
          to warn up, will be as cosine lr scheduler
        - eta_min: float, setup ConsinAnnealingLR eta_min while warm_up_epochs = 0

    Returns:
        - scheduler
    """

    if warm_up_epochs <= 0:
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
        warm_up_with_cosine_lr =lambda epoch: 0.5 * (
                np.cos((epoch) / (epochs) * np.pi) + 1)
        scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    else:
        warm_up_with_cosine_lr = lambda epoch: eta_min + (
                    epoch / warm_up_epochs) if epoch <= warm_up_epochs else 0.5 * (
                np.cos((epoch - warm_up_epochs) / (epochs - warm_up_epochs) * np.pi) + 1)
        scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    return scheduler


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 使用He初始化
def hekaiming_init(m):
    if isinstance(m, torch.nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, torch.nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

def main():
    seed = 42
    setup_seed(seed)
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    if "consistency" in args.training_mode:
        distillation = True
    else:
        distillation = False

    logger.log("creating model and diffusion...")


    #load consistency model
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


    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None


    generator = get_generator(args.generator, args.num_samples, args.seed)

    #load data

    all_files = _list_image_files_recursively(args.data_dir)
    all_iv_files = _list_image_files_recursively(args.image_iv_paths)
    traindataset = ImageDataset(
        data_dir=args.data_dir,
        data_ir_dir=args.image_iv_paths,
        resolution=160,
        image_paths=all_files,
        image_iv_paths=all_iv_files,
        random_crop=True,
        random_flip=True,
        isCrop=False,
        istrain=False
    )
    trainloader = DataLoader(
        traindataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    #fusion_model_save_path
    weight_path = '../model_weights/without_consitency_fusion_head_with_att_directimg_new'
    os.makedirs(weight_path,exist_ok=True)


    #tensorboard view
    writer = SummaryWriter(log_dir='../scalar/without_consitency_fusion_head_with_att_directimg_new')


    #epoch
    epoch= 6
    model_kwargs = {}

    # add_noise  40 mean min noise adding to image   0 mean max noise adding to image
    noise_step = 40

    # create fusion model
    fusion_model = Fusion_Head().to(dist_util.dev())


    # opitimizer
    opitimizer = th.optim.AdamW(fusion_model.parameters(), lr=1e-4, weight_decay=1e-5)

    #scheduler
    scheduler = th.optim.lr_scheduler.StepLR(opitimizer, 2, gamma=0.1)

    #loss
    loss = Fusion_loss().to(dist_util.dev())

    #iter
    iter = 0

    logger.log("training...")
    for i in range(0,epoch):

        print("第{}次迭代开始".format(i+1))
        sum_loss_in = 0
        sum_loss_grad = 0
        sum_loss_fs = 0

        loop = tqdm(enumerate(trainloader), total=len(trainloader))
        for j, (batch, path) in loop:

            iter = iter + 1
            fd = []
            fs = []
            batch['img'] = batch['img'].to(dist_util.dev())
            batch['vis'] = batch['vis'].to(dist_util.dev())
            batch['ir'] = batch['ir'].to(dist_util.dev())
            model.eval()
            with th.no_grad():
                    # random_number = random.randint(20, 40)
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

                        fs.append(feat)
                        # fd.append(feat[0])
                        del feat
            opitimizer.zero_grad()  # 梯度降零


            If = fusion_model(fs[0])
            If = (If + 1)/2
            batch['vis'] = (batch['vis'] + 1)/2
            batch['ir'] = (batch['ir'] + 1) / 2
            Ir = batch['vis']+batch['ir']-If


            del fd
            loss_in, loss_grad = loss(batch['vis'], batch['ir'], If , Ir)

            loss_fs = loss_in + loss_grad
            loss_fs.backward()
            opitimizer.step()
            sum_loss_grad = sum_loss_grad + loss_grad.item()
            sum_loss_in = sum_loss_in + loss_in.item()
            sum_loss_fs = sum_loss_fs + loss_fs.item()

            loop.set_description(f'Epoch [{i + 1}/{epoch}]')
            loop.set_postfix(loss_fs=sum_loss_fs / (j+1))

            if (iter + 1) % args.log_interval ==0:
                 writer.add_scalar("train_loss_in", sum_loss_in/(j+1), iter/100)
                 writer.add_scalar("train_loss_grad", sum_loss_grad/(j+1), iter/100)
                 writer.add_scalar("train_loss_fs", sum_loss_fs/(j+1), iter/100)


            if (iter + 1) % args.save_interval == 0:
                all_states = {"fusion_model": fusion_model.state_dict(), "opitimizer": opitimizer.state_dict(),
                              "epoch": i + 1}
                th.save(obj=all_states,
                        f="../model_weights/without_consitency_fusion_head_with_att_directimg_new/all_states_{}.pth".format(int((iter+1) / 2000)))

        scheduler.step()

    all_states = {"fusion_model": fusion_model.state_dict(), "opitimizer": opitimizer.state_dict(),
                      "epoch": i + 1}
    th.save(obj=all_states,
                f="../model_weights/without_consitency_fusion_head_with_att_directimg_new/all_states.pth")



    logger.log("training complete")


def create_argparser():
    defaults = dict(
        training_mode="consistency_training",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        batch_size=10,
        sampler="onestep",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="../model_weights/cmt1.pt",
        log_interval=200,
        save_interval=2000,
        seed=42,
        ts="",
        data_dir='../../../../../Disk_B/datasets/imagefusion/KAIST-database/visible',
        image_iv_paths='../../../../../Disk_B/datasets/imagefusion/KAIST-database/lwir',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
