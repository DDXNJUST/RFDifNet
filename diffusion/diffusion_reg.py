import math
import pdb

import torch
from torch import nn
from inspect import isfunction
from functools import partial
import numpy as np
# from . import loss
from . import loss_reg as loss
from tqdm import tqdm
from torch import Tensor
from pathlib import Path
import cv2

def save_flow_2(flow: [Tensor], dst, im_name: str = '', step=32):
    # pdb.set_trace()
    flow = flow[-1, :, :, :].permute(1, 2, 0) # * 255.
    flow = flow.detach().cpu().numpy()
    image = np.full((256, 256, 3), 255.0)  # image = np.ones([256, 256, 3])
    h, w = image.shape[:2]  # h w 3
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T  # h w 2

    lines = np.vstack([x, y, x + fx * 6., y + fy * 6.]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(image, (x1, y1), (x2, y2), (0, 0, 0), 1, tipLength=0.2)
    cv2.imwrite(str(dst)+im_name, image)

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) /
                n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    def repeat_noise(): return torch.randn(
        (1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))

    def noise(): return torch.randn(shape, device=device)

    return repeat_noise() if repeat else noise()


class GaussianBlurConv(nn.Module):
    def __init__(self, channels):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = [[0.0265, 0.0354, 0.0390, 0.0354, 0.0265],
                  [0.0354, 0.0473, 0.0520, 0.0473, 0.0354],
                  [0.0390, 0.0520, 0.0573, 0.0520, 0.0390],
                  [0.0354, 0.0473, 0.0520, 0.0473, 0.0354],
                  [0.0265, 0.0354, 0.0390, 0.0354, 0.0265]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        # pdb.set_trace()
        x = torch.nn.functional.conv2d(x, self.weight, padding=2, groups=self.channels)
        return x


class GaussianDiffusion_reg(nn.Module):
    def __init__(
            self,
            denoise_fn_reg,
            stn,
            channels=3,
            loss_type='l1',
            conditional=True,
            schedule_opt=None,
            loss_lambda=1,
            gamma=1
    ):
        super().__init__()
        self.channels = channels
        self.denoise_fn_reg = denoise_fn_reg
        self.stn = stn
        self.conditional = conditional
        self.loss_type = loss_type
        self.lambda_L = loss_lambda
        self.gamma = gamma
        if schedule_opt is not None:
            pass

        self.loss_func = nn.MSELoss(reduction='mean').to('cuda:0')
        self.loss_ncc = loss.crossCorrelation3D(1, kernel=(9, 9), gamma=self.gamma).to('cuda:0')
        self.loss_reg = loss.gradientLoss("l2").to('cuda:0')
        # self.loss_grad = nn.L1Loss(reduction='mean').to('cuda:0')
        self.gaussian_conv_1 = GaussianBlurConv(1).to('cuda:0')
        self.loss_grad = nn.L1Loss(reduction='mean').to('cuda:0')
        # print('gamma', self.gamma)

        self.clamp_range = (0, 1)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='mean').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='mean').to(device)
        else:
            raise NotImplementedError()
        # self.loss_ncc = loss.crossCorrelation3D(1, kernel=(9, 9, 9),gamma=self.gamma).to(device)
        # self.loss_ncc = loss.crossCorrelation3D(1, kernel=(9, 9), gamma=self.gamma).to(device)
        self.loss_ncc = loss.crossCorrelation3D(4, kernel=(9, 9), gamma=self.gamma).to(device)
        self.loss_reg = loss.gradientLoss("l2").to(device)

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        # pdb.set_trace()
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
                             (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        # pdb.set_trace()
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        b=x.shape[0]
        if condition_x is not None:
            with torch.no_grad():
                # score = self.denoise_fn_reg(torch.cat([condition_x, x], dim=1), t)
                # _, score = self.denoise_fn_reg(
                #     torch.cat([condition_x['M'], condition_x['L'].view((b, 4, 32, 256, 256)).mean(dim=2), x], dim=1), condition_x['M'], t)
                _, score = self.denoise_fn_reg(x, torch.cat([condition_x['M'], condition_x['L'].view((b, condition_x['M'].size(1), int(condition_x['L'].size(1)/condition_x['M'].size(1)), 256, 256)).mean(dim=2)],dim=1),t)

            # pdb.set_trace()
            # x_recon = self.predict_start_from_noise(
            #     x, t=t, noise=score)
            x_recon = score
        # else:
        #     x_recon = self.predict_start_from_noise(
        #         x, t=t, noise=self.denoise_fn_reg(x, t))

        # if clip_denoised:
        #     x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample_loop(self, x_in):
        # pdb.set_trace()
        clip_noise = True if exists(self.clamp_range) else False
        sample_inter = 1 | (self.num_timesteps // 10)
        device = self.betas.device
        b, c, h, w = x_in['M'].shape
        shape = x_in['M'].shape[2:]
        # x_noisy_fw_reg = torch.randn((b, 4, *shape), device=self.betas.device)
        # with torch.no_grad():
        #     t = torch.full((b,), 0, device=device, dtype=torch.long)
        #     _, flow = self.denoise_fn_reg(torch.cat([x_in['M'], x_in['L'].view((b, 4, 32, 256, 256)).mean(dim=2), x_noisy_fw_reg], dim=1), x_in['M'], t)
        #     deform = self.stn(x_in['M'], flow)
        #     return deform, flow

        img = torch.randn((b, 2, *shape), device=self.betas.device)
        # img = torch.zeros(b, 2, w, h).cuda()
        save_flow_2(img, './disp/', '0_2.jpg')
        for i in tqdm(
                reversed(range(0, 25)),  #self.num_timesteps)),
                desc="sampling loop time step",
                total=25,
        ):
            # self_cond = x_start if self.self_condition else None
            # img = self.p_sample(
            #     img,
            #     torch.full((b,), i, device=device, dtype=torch.long),
            #     condition_x=x_in,
            #     self_cond=self_cond,
            #     clip_denoised=clip_noise,
            #     get_interm_fm=get_interm_fm,
            # )
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(self, img, t, repeat_noise=False, condition_x=x_in)
            # x_start = img
            if (i+1) % 100 == 0:
                save_flow_2(img, './disp/', str(i)+'_2.jpg')
        out_img = self.stn(x_in['M'], img)
        save_flow_2(img, './disp/', str(i) + 'out.jpg')
        return out_img, img

    def ddim_sample_loop(self, x_in, section_counts="ddim300", eta=0.0):  # torch.Size([1, 132, 256, 256]) 'ddim25'
        pdb.set_trace()
        use_timesteps = self.space_timesteps(self.num_timesteps, section_counts)  # len=25 (500, 25)
        self.space_new_betas(use_timesteps)

        pred_x_start = None
        if not self.conditional:
            assert isinstance(x_in, [list, tuple])
            shape = x_in
            b = shape[0]
            img = torch.randn(shape, device=self.betas.device)
            for i in tqdm(
                reversed(range(0, len(self.betas))),
                desc="ddim sampling loop time step",
                total=len(self.betas),
            ):
                self_cond = pred_x_start if self.self_condition else None
                img = self.ddim_sample(
                    img,
                    torch.full((b,), i, device=self.betas.device, dtype=torch.long),
                    self_cond=self_cond,
                    eta=eta,
                )
            return img
        else:  #
            assert isinstance(x_in, torch.Tensor)
            x = x_in  # torch.Size([1, 132, 256, 256])
            shape = x.shape[2:]  # torch.Size([256, 256])
            b = x.shape[0]  # 1
            img = torch.randn((b, self.channels, *shape), device=self.betas.device)  # torch.Size([1, 128, 256, 256])
            for i in tqdm(
                reversed(range(0, len(self.betas))),  # tensor([8.7440e-05], device='cuda:0')
                desc="ddim sampling loop time step",
                total=len(self.betas),
            ):
                self_cond = pred_x_start if self.self_condition else None  # None
                img = self.ddim_sample(
                    img,  # torch.Size([1, 128, 256, 256])
                    torch.full((b,), i, device=x.device, dtype=torch.long),  # tensor([0], device='cuda:0')
                    condition_x=x,  # torch.Size([1, 132, 256, 256])
                    self_cond=self_cond,  # none
                    eta=eta,  # eta 0.0
                )
            return img

    def p_sample(self, model, x, t, clip_denoised=True, repeat_noise=False, condition_x=None):
        # pdb.set_trace()
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def registration(self, x_in):
        return self.p_sample_loop(x_in)

    def q_sample_reg(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        # fix gama
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_in, noise=None):
        [b, c, h, w] = x_in['L'].shape
        x_start = torch.zeros(b, 2, w, h).cuda()
        t = torch.randint(0, self.num_timesteps, (b,), device=x_start.device).long()
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy_fw_reg = self.q_sample_reg(x_start=x_start, t=t, noise=noise)
        x_recon_fw, flow_fw = self.denoise_fn_reg(x_noisy_fw_reg, torch.cat([x_in['M'],x_in['L'].view(b, x_in['M'].size(1), int(x_in['L'].size(1)/x_in['M'].size(1)), 256, 256).mean(dim=2)],dim=1),t)
        output_fw = self.stn(x_in['M'], flow_fw)
        l_pix_fw = self.loss_func(noise, x_recon_fw)
        l_sim_fw = self.loss_ncc(output_fw[:, 0:1, :, :], x_in['F'][:, 0:1, :, :], x_recon_fw[:, 0:1, :, :]) * 200  # 20
        l_smt_fw = self.loss_reg(flow_fw) * 20
        loss = l_smt_fw + l_pix_fw + l_sim_fw
        return [x_recon_fw, output_fw, flow_fw], [l_pix_fw, l_sim_fw, l_smt_fw, loss]

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
