import pdb
from functools import partial
import time
from copy import deepcopy

import einops
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torchvision as tv
from scipy.io import savemat
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset.pan_dataset import PanDataset
from dataset.hisr import HISRDataSets
from diffusion.diffusion_ddpm_pan import make_beta_schedule
from utils.logger import TensorboardLogger
from utils.lr_scheduler import get_lr_from_optimizer, StepsAll
from utils.metric import AnalysisPanAcc, NonAnalysisPanAcc
from utils.misc import compute_iters, exist, grad_clip, model_load, path_legal_checker
from utils.optim_utils import EmaUpdater

import skimage.io
import os
from torch.nn import init

from collections import OrderedDict
from datetime import datetime

from diffusion import diffusion_reg
from models import unet_reg_fix as unet_reg  # unet_reg_fix

from models.sr3_test_fix import UNetSR3 as Unet  # sr3_test_fix
from diffusion.diffusion_ddpm_pan import GaussianDiffusion

from diffusion import loss_reg

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def show_distribution(hr):
    plt.figure()
    gt = hr[:2].detach().cpu().flatten().numpy()
    sns.displot(data=gt)
    plt.show()


def norm(x):
    # x range is [0, 1]
    return x * 2 - 1


def unorm(x):
    # x range is [-1, 1]
    return (x + 1) / 2


def clamp_fn(g_sr):
    def inner(x):
        x = x + g_sr
        x = x.clamp(0, 1.0)
        return x - g_sr

    return inner

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    net.apply(weights_init_orthogonal)

def engine_google(
    # dataset
    train_dataset_path,
    valid_dataset_path,
    dataset_name=None,
    # image settings
    image_n_channel=8,
    image_size=64,
    # diffusion settings
    schedule_type="cosine",
    n_steps=3_000,
    max_iterations=400_000,
    # device setting
    device="cuda:0",
    # optimizer settings
    batch_size=128,
    lr_d=1e-4,
    show_recon=False,
    # pretrain settings
    pretrain_weight=None,
    pretrain_iterations=None,
    *,
    # just for debugging
    constrain_channel=None,
):
    # init logger
    stf_time = time.strftime("%m-%d_%H-%M", time.localtime())
    comment = "pandiff"
    logger = TensorboardLogger(file_logger_name="{}-{}".format(stf_time, comment))

    dataset_name = (
        train_dataset_path.strip(".h5").split("_")[-1]
        if not exist(dataset_name)
        else dataset_name
    )
    logger.print(f"dataset name: {dataset_name}")
    division_dict = {"Geo": 2047.0, "Chikusei": 5000.0, "Pavia": 8000.0}
    logger.print(f"dataset norm division: {division_dict[dataset_name]}")

    if dataset_name in ['Pavia']:
        image_n_channel = 102
        image_size = 256
        add_n_channel = 1
        inner_channel = 32
    elif dataset_name in ['Chikusei']:
        image_size = 256
        image_n_channel = 128
        add_n_channel = 4
        inner_channel = 64
    elif dataset_name in ['Geo']:
        image_size = 256
        image_n_channel = 4
        add_n_channel = 1
        inner_channel = 32

    # get dataset
    d_train = h5py.File(train_dataset_path)
    d_valid = h5py.File(valid_dataset_path)
    DatasetUsed = partial(
        PanDataset,
        full_res=False,
        norm_range=False,
        constrain_channel=constrain_channel,
        division=division_dict[dataset_name],
        aug_prob=0,
        wavelets=False,
    )

    ds_train = DatasetUsed(
        d_train,
    )
    ds_valid = DatasetUsed(
        d_valid,
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=16,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
    )

    ###### 融合设置 ######
    # initialize models
    torch.cuda.set_device(device)
    denoise_fn = Unet(
        in_channel=image_n_channel,
        out_channel=image_n_channel,
        lms_channel=image_n_channel,
        pan_channel=add_n_channel,
        inner_channel=inner_channel,
        norm_groups=1,
        channel_mults=(1, 2, 2, 4),
        attn_res=(8,),
        dropout=0.2,
        image_size=64,
        self_condition=True,
    ).to(device)

    # diffusion
    diffusion = GaussianDiffusion(
        denoise_fn,
        image_size=image_size,
        channels=image_n_channel,
        # num_sample_steps=n_steps,
        pred_mode="x_start",  #  noise
        loss_type="l2",  # l1
        device=device,
        clamp_range=(0, 1),
    )
    diffusion.set_new_noise_schedule(
        betas=make_beta_schedule(schedule="cosine", n_timestep=n_steps, cosine_s=8e-3)
    )  # 定义好的噪声、α、β等
    diffusion = diffusion.to(device)

    # model, optimizer and lr scheduler
    diffusion_dp = (
        diffusion
    )
    ema_updater = EmaUpdater(
        diffusion_dp, deepcopy(diffusion_dp), decay=0.995, start_iter=20_000
    )
    opt_d = torch.optim.AdamW(denoise_fn.parameters(), lr=lr_d, weight_decay=1e-4)

    scheduler_d = torch.optim.lr_scheduler.MultiStepLR(
        opt_d, milestones=[100_000, 200_000, 350_000], gamma=0.2
    )
    schedulers = StepsAll(scheduler_d)

    model_score = unet_reg.UNet(
        in_channel=2,  # model_opt['unet']['in_channel'],  # 12 3
        out_channel=2,  # model_opt['unet']['out_channel'],  # 4 1
        inner_channel=16,  # model_opt['unet']['inner_channel'],  # 8 16
        channel_mults=[1, 2, 3, 4],  # model_opt['unet']['channel_multiplier'],  # [1, 2, 4, 4] [1, 2, 3, 4]
        attn_res=[10],  # model_opt['unet']['attn_res'],  # [10]
        res_blocks=1,  # model_opt['unet']['res_blocks'],  # 1
        dropout=0,  # model_opt['unet']['dropout'],  # 0
        image_size=[128, 128, 32],  # model_opt['diffusion']['image_size'],  # [128, 128, 32]
        # opt=self_opt  # None
    )

    stn = unet_reg.Dense3DSpatialTransformer()
    netG_reg = diffusion_reg.GaussianDiffusion_reg(
        model_score, stn,
        channels=None,  # model_opt['diffusion']['channels'],  # None
        loss_type='l2',  # L1 or L2
        conditional=True,  # model_opt['diffusion']['conditional'],  # True
        schedule_opt={'schedule': 'linear', 'n_timestep': n_steps, 'linear_start': 1e-06, 'linear_end': 0.01},
        loss_lambda=20,  # model_opt['loss_lambda'],  # 20
        gamma=1,  # model_opt['gamma'],  # 1
    ).to(torch.device('cuda:0'))

    init_weights(netG_reg.denoise_fn_reg, init_type='orthogonal')
    init_weights(netG_reg.stn, init_type='normal')

    print("Model Initialized")
    path_checkpoint = os.path.join('experiments', 'FSDiffReg_{}'.format(datetime.now().strftime('%y%m%d_%H%M%S')))

    total_params = sum(p.numel() for p in denoise_fn.parameters())
    print(f"Total parameters: {total_params}")
    # for p in denoise_fn.parameters():
    #     print(p.dtype)
    total_params = sum(p.numel() for p in model_score.parameters())
    print(f"Total parameters: {total_params}")

    if isinstance(path_checkpoint, str):
        os.makedirs(path_checkpoint, exist_ok=True)

    log_loss_min = 100
    log_epoch_loss_min = 0
    log_NCC_max = 0
    log_NCC_epoch_max = 0

    optG_reg = torch.optim.Adam(list(netG_reg.parameters()), lr=0.0002, betas=(0.5, 0.999))
    netG_reg.set_new_noise_schedule({'schedule': 'linear', 'n_timestep': n_steps, 'linear_start': 1e-06, 'linear_end': 0.01}, torch.device('cuda:0'))  # 2000
    criterion = loss_reg.FractionalDerivativeLoss(alpha=0.5, kernel_size=30).to(torch.device('cuda:0'))

    iterations = 0
    while iterations <= max_iterations:
        for i, (pan, lms, hr, unalignedpan) in enumerate(dl_train, 1):
            pan, lms, hr, unalignedpan = map(lambda x: x.cuda(), (pan, lms, hr, unalignedpan))
            lms = torch.nn.functional.interpolate(lms, size=(256, 256), mode='bilinear')
            hr_ca = hr.view(1, add_n_channel, int(image_n_channel/add_n_channel), 256, 256).mean(dim=2)
            optG_reg.zero_grad()
            result_reg, loss_reg_ = netG_reg({'F': hr_ca.to(torch.device('cuda:0')), 'M': unalignedpan.to(torch.device('cuda:0')),
                                              'L': lms.to(torch.device('cuda:0'))})
            x_recon_fw, out_M, _ = result_reg
            l_pix, l_sim, l_smt, l_tot = loss_reg_
            # ###########################
            if iterations > 0:  # 10000
                _, recon_x_before = diffusion_dp(out_M.repeat(1, int(image_n_channel/add_n_channel), 1, 1) - lms, cond=einops.pack([lms, out_M, ], "b * h w", )[0], ref=hr - lms)
                recon_x_before_ca = recon_x_before.view(1, add_n_channel, int(image_n_channel/add_n_channel), 256, 256).mean(dim=2)
                loss_joint = 0.001 * criterion(recon_x_before_ca, (hr_ca - out_M))
            else:
                loss_joint = 0.0
            # ###########################
            # loss_joint = 0.0
            (l_tot+loss_joint).backward(retain_graph=True)
            cond, _ = einops.pack([lms, out_M, ], "b * h w", )
            res = out_M.repeat(1, int(image_n_channel/add_n_channel), 1, 1) - lms
            ref = hr - lms
            diff_loss, recon_x = diffusion_dp(res, cond=cond, ref=ref)
            opt_d.zero_grad()
            diff_loss.backward()
            recon_x = recon_x.clone() + lms
            grad_clip(diffusion_dp.model.parameters(), mode="norm", value=0.003)
            optG_reg.step()
            opt_d.step()
            ema_updater.update(iterations)
            schedulers.step()

            iterations += 1
            logger.print(
                f"[iter {iterations}/{max_iterations}: "
                + f"d_lr {get_lr_from_optimizer(opt_d): .6f}] - "
                + f"denoise loss {diff_loss:.6f} "
                + f"l_pix {l_pix:.6f} "
                + f"l_sim {l_sim:.6f} "
                + f"l_smt {l_smt:.6f} "
                + f"l_tot {l_tot:.6f} "
                + f"loss_joint {loss_joint:.6f} "
            )

            # test predicted sr
            if show_recon and iterations % 1_000 == 0:
                # NOTE: only used to validate code

                print('log_loss_min:', log_loss_min, 'log_epoch_min:', log_epoch_loss_min)
                logger.print(
                    f"[iter {iterations}/{max_iterations}: "
                    + f"log_loss_min {log_loss_min:.6f} "
                    + f"log_epoch_loss_min {log_epoch_loss_min:.6f} "
                )
                for i, (pan_test, lms_test, hr_test, unalignedpan_test) in enumerate(dl_valid, 1):
                    pan_test, lms_test, hr_test, unalignedpan_test = map(lambda x: x.cuda(),
                                                                  (pan_test, lms_test, hr_test, unalignedpan_test))
                    lms_test = torch.nn.functional.interpolate(lms_test, size=(256, 256), mode='bilinear')
                    hr_ca_test = hr_test.view(10, add_n_channel, int(image_n_channel/add_n_channel), 256, 256).mean(dim=2)
                    test_data_reg = {'F': hr_ca_test.to(torch.device('cuda:0')), 'M': unalignedpan_test.to(torch.device('cuda:0')),
                                     'L': lms_test.to(torch.device('cuda:0'))}
                    out_M_test, _ = netG_reg.registration(test_data_reg)
                    mse = loss_reg.mse_loss(pan_test, out_M_test)
                    ncc = loss_reg.normalized_cross_correlation(pan_test, out_M_test)
                    mi = loss_reg.mutual_information(pan_test, out_M_test)
                    print(f'MSE: {mse.item()}, NCC: {ncc.mean().item()}, MI: {mi.item()}')
                    logger.print(
                        f"[iter {iterations}/{max_iterations}: "
                        + f"MSE: {mse.item()} "
                        + f"NCC: {ncc.mean().item()} "
                        + f"MI: {mi.item()}"
                    )
                    if ncc.mean().item() > log_NCC_max:
                        log_NCC_max = ncc.mean().item()
                        log_NCC_epoch_max = iterations
                    print('log_NCC_max:', log_NCC_max, 'log_NCC_epoch_max:', log_NCC_epoch_max)
                    logger.print(
                        f"[iter {iterations}/{max_iterations}: "
                        + f"log_NCC_max {log_NCC_max:.6f} "
                        + f"log_NCC_epoch_max {log_NCC_epoch_max:.6f} "
                    )

            if iterations % 5_000 == 0:
                genG_path = os.path.join(
                    path_checkpoint, 'I{}_gen_G.pth'.format(iterations))
                network = netG_reg
                state_dict = network.state_dict()
                for key, param in state_dict.items():
                    state_dict[key] = param.cpu()
                torch.save(state_dict, genG_path)
            # do some sampling to check quality
                diffusion_dp.model.eval()
                ema_updater.ema_model.model.eval()
                netG_reg.eval()

                analysis_d = AnalysisPanAcc()
                with torch.no_grad():
                    for i, (pan, lms, hr, unalignedpan) in enumerate(dl_valid, 1):
                        torch.cuda.empty_cache()
                        pan_test, lms_test, hr_test, unalignedpan_test = map(
                            lambda x: x.cuda(), (pan, lms, hr, unalignedpan)
                        )
                        lms_test = torch.nn.functional.interpolate(lms_test, size=(256, 256), mode='bilinear')
                        hr_ca_test = hr_test.view(10, add_n_channel, int(image_n_channel/add_n_channel), 256, 256).mean(dim=2)
                        test_data_reg = {'F': hr_ca_test.to(torch.device('cuda:0')),
                                          'M': unalignedpan_test.to(torch.device('cuda:0')),
                                       'L': lms_test.to(torch.device('cuda:0'))}
                        out_M_test, flow = netG_reg.registration(test_data_reg)
                        cond_test, _ = einops.pack(
                            [
                                lms_test,
                                out_M_test,
                            ],
                            "b * h w",
                        )

                        sr_test = ema_updater.ema_model(cond_test, mode="ddim_sample", section_counts="ddim25")
                        sr_test = sr_test + lms_test
                        sr_test = sr_test.clip(0, 1)

                        hr_test = hr_test.to(sr_test.device)
                        analysis_d(hr_test, sr_test)
                        

                        logger.print("---diffusion result---")
                        logger.print(analysis_d.last_acc)
                    if i != 1:
                        logger.print("---diffusion result---")
                        logger.print(analysis_d.print_str())
                diffusion_dp.model.train()
                setattr(ema_updater.model, "image_size", 64)
                if iterations % 5_000 == 0:
                    torch.save(
                        ema_updater.on_fly_model_state_dict,
                        f"./Fus_experiments/diffusion_{dataset_name}_iter_{iterations}.pth",
                    )
                    torch.save(
                        ema_updater.ema_model_state_dict,
                        f"./Rus_experiments/ema_diffusion_{dataset_name}_iter_{iterations}.pth",
                    )
                    logger.print("save model")

                logger.log_scalars("diffusion_perf", analysis_d.acc_ave, iterations)
                logger.print("saved performances")

            # log loss
            if iterations % 50 == 0:
                logger.log_scalar("denoised_loss", diff_loss.item(), iterations)

torch.cuda.set_device(0)
engine_google(
    # "YOUR DATA PATH HERE",
    train_dataset_path="/home/dwx/dwx/code/Dif-PAN+FSDiffReg_HSI/data/Chikusei_unaligned/ChikuseiTrain.h5",
    valid_dataset_path="/home/dwx/dwx/code/Dif-PAN+FSDiffReg_HSI/data/Chikusei_unaligned/ChikuseiVal.h5",
    dataset_name="Chikusei",
    show_recon=True,
    lr_d=1e-4,
    n_steps=500,
    schedule_type="cosine",
    batch_size=1,
    device="cuda:0",
    max_iterations=300_000,
    image_n_channel=128,  # 102,  # 128,  # 8
    # pretrain_iterations=0,
)