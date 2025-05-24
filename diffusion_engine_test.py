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
from utils.metric import AnalysisPanAcc, NonAnalysisPanAcc
from utils.misc import compute_iters, exist, grad_clip, model_load, path_legal_checker

from torch.nn import init
import scipy.io as scio


from diffusion import diffusion_reg
from models import unet_reg_fix as unet_reg

from models.sr3_test_fix import UNetSR3 as Unet
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

# import cudnn
@torch.no_grad()
def test_fn(
    test_data_path,
    weight_path,
    schedule_type="cosine",
    batch_size=320,
    n_steps=1500,
    show=False,
    device="cuda:0",
    full_res=False,
    dataset_name="Geo",
    division=8000.0  # 5000.0,
):

    torch.cuda.set_device(device)
    # load model
    if dataset_name in ['Pavia']:
        image_n_channel = 102
        image_size = 256
        pan_channel = 1
        inner_channel = 32
    elif dataset_name in ['Chikusei']:
        image_size = 256
        image_n_channel = 128
        pan_channel = 4
        inner_channel = 64
    elif dataset_name in ['Geo']:
        image_size = 256
        image_n_channel = 4
        pan_channel = 1
        inner_channel = 32
    denoise_fn = Unet(
        in_channel=image_n_channel,
        out_channel=image_n_channel,
        lms_channel=image_n_channel,
        pan_channel=pan_channel,#1,
        inner_channel=inner_channel,  # 64,  # 32
        norm_groups=1,
        channel_mults=(1, 2, 2, 4),  # (64, 32, 16, 8)
        attn_res=(8,),
        dropout=0.2,
        image_size=64,
        self_condition=True,
    ).to(device)
    denoise_fn = model_load(weight_path, denoise_fn, device=device)

    denoise_fn.eval()
    print(f"load weight {weight_path}")
    diffusion = GaussianDiffusion(
        denoise_fn,
        image_size=image_size,
        channels=image_n_channel,
        pred_mode="x_start",
        loss_type="l1",
        device=device,
        clamp_range=(0, 1),
    )
    diffusion.set_new_noise_schedule(
        betas=make_beta_schedule(schedule="cosine", n_timestep=n_steps, cosine_s=8e-3)
    )
    diffusion = diffusion.to(device)

    model_score = unet_reg.UNet(
        in_channel=2,  # model_opt['unet']['in_channel'],  # 3  12
        out_channel=2,  # model_opt['unet']['out_channel'],  # 1  4
        inner_channel=16,  # model_opt['unet']['inner_channel'],  # 8 16
        channel_mults=[1, 2, 3, 4],  # model_opt['unet']['channel_multiplier'],  # [1, 2, 3, 4] [1, 2, 4, 4]
        attn_res=[10],  # model_opt['unet']['attn_res'],  # [10] 8
        res_blocks=1,  # model_opt['unet']['res_blocks'],  # 1
        dropout=0,  # model_opt['unet']['dropout'],  # 0
        image_size=[128, 128, 32],  # model_opt['diffusion']['image_size'],  # [128, 128, 32]
        # opt=self_opt  # None
    )
    model_score.eval()
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
    netG_reg.set_new_noise_schedule(
        {'schedule': 'linear', 'n_timestep': n_steps, 'linear_start': 1e-06, 'linear_end': 0.01}, torch.device('cuda:0'))
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    netG_reg.load_state_dict(
        torch.load('/home/dwx/dwx/code/Dif-PAN+FSDiffReg_reg/experiments/Chikusei_2000_FSDiffReg_250422_031657/I300000_gen_G.pth'), strict = True)

    # load dataset
    d_test = h5py.File(test_data_path)
    if dataset_name in ["Geo", "Pavia", "Chikusei"]:
        ds_test = PanDataset(
            d_test, full_res=full_res, norm_range=False, division=division, wavelets=False
        )
    else:
        ds_test = HISRDataSets(
            d_test, normalize=False, aug_prob=0.0, wavelets=True
        )
        
    dl_test = DataLoader(
            ds_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0
        )

    # saved_name = "reduced" if not full_res else "full"

    # do sampling
    # preds = []
    sample_times = len(dl_test)
    analysis = AnalysisPanAcc() if not full_res else NonAnalysisPanAcc()
    total_time_reg = 0
    total_time_fus = 0
    for i, (pan, lms, hr, unalignedpan) in enumerate(dl_test, 1):
        # pdb.set_trace()
        print(f"sampling [{i}/{sample_times}]")
        pan, lms, hr, unalignedpan = map(lambda x: x.cuda(), (pan, lms, hr, unalignedpan))
        lms = torch.nn.functional.interpolate(lms, size=(256, 256), mode='bilinear')
        hr_ca = hr.view(1, pan_channel, int(image_n_channel/pan_channel), 256, 256).mean(dim=2)
        start_time_reg = time.time()
        test_data_reg = {'F': hr_ca.to(torch.device('cuda:0')),
                         'M': unalignedpan.to(torch.device('cuda:0')),
                         'L': lms.to(torch.device('cuda:0'))}
        out_M, flow = netG_reg.registration(test_data_reg)
        end_time_reg = time.time()
        total_time_reg += (end_time_reg - start_time_reg)
        cond, _ = einops.pack(
            [lms, out_M],
            "b * h w",
        )
        start_time_fus = time.time()
        sr = diffusion(cond, mode="ddim_sample", section_counts="ddim25")
        end_time_fus = time.time()
        total_time_fus += (end_time_fus - start_time_fus)

        sr = sr + lms.cuda()
        sr = sr.clip(0, 1)
        analysis(sr.detach().cpu(), hr.cpu())

        print(analysis.print_str(analysis.last_acc))

        # pdb.set_trace()
        img_fusion = (sr * division).to(torch.device('cpu')).numpy()
        img_fusion = img_fusion[-1, :, :, :].transpose(1, 2, 0)
        # skimage.io.imsave(f'./result_images/fusion/fusion_{i}.tif', img_fusion.astype(np.uint16))
        scio.savemat(f'./result_images/fusion/fusion_{i}.mat', {'fusion': img_fusion})
        img_Reg = (out_M * division).to(torch.device('cpu')).numpy()
        img_Reg = img_Reg[-1, :, :, :].transpose(1, 2, 0)
        # skimage.io.imsave(f'./result_images/Reg/Reg_{i}.tif', img_Reg.astype(np.uint16))
        scio.savemat(f'./result_images/Reg/Reg_{i}.mat', {'Reg': img_Reg})
        hr = (hr * division).to(torch.device('cpu')).numpy()
        hr = hr[-1, :, :, :].transpose(1, 2, 0)
        # skimage.io.imsave(f'./result_images/gt/gt_{i}.tif', hr.astype(np.uint16))
        scio.savemat(f'./result_images/gt/gt_{i}.mat', {'gt': hr})
        pan = (pan * division).to(torch.device('cpu')).numpy()
        pan = pan[-1, :, :, :].transpose(1, 2, 0)
        # skimage.io.imsave(f'./result_images/pan/pan_{i}.tif', pan.astype(np.uint16))
        scio.savemat(f'./result_images/pan/pan_{i}.mat', {'pan': pan})

        print(f"over all test acc:\n {analysis.print_str()}")

    print("save result")
    print('total_time_reg', total_time_reg)
    print('total_time_fus', total_time_fus)

torch.cuda.set_device(0)
test_fn(
    # "YOUR DATA PATH HERE",
    test_data_path="/home/dwx/dwx/code/Dif-PAN+FSDiffReg_HSI/data/Chikusei_unaligned/ChikuseiTest.h5",
    weight_path="/home/dwx/dwx/code/Dif-PAN+FSDiffReg_HSI/weights/chikusei_fix/ema_diffusion_Geo_iter_300000.pth",
    batch_size=1,
    n_steps=2000,
    show=True,
    dataset_name="Chikusei",
    division=5000.0,
    full_res=False,
    device="cuda:0",
)
