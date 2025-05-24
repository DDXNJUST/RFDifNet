import math
import pdb

import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from einops import rearrange

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# model
class TimeEmbedding_reg(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample_reg(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        # self.conv = nn.Conv3d(dim, dim, 3, padding=1)
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample_reg(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.conv = nn.Conv3d(dim, dim, 3, 2, 1)
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block_reg(nn.Module):
    def __init__(self, dim, dim_out, groups=4, dropout=0):
        super().__init__()
        self.block_reg = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            # nn.Conv3d(dim, dim_out, 3, padding=1)
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        # pdb.set_trace()
        return self.block_reg(x)


class ResnetBlock_reg(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, dropout=0):
        super().__init__()
        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None

        self.block1 = Block_reg(dim, dim_out)
        self.block2 = Block_reg(dim_out, dim_out, dropout=dropout)
        # self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        # pdb.set_trace()
        h = self.block1(x)
        if exists(self.mlp):
            # h += self.mlp(time_emb)[:, :, None, None, None]
            h = h + self.mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention_reg(nn.Module):
    def __init__(self, in_channel, n_head=4):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(4, in_channel)
        # self.qkv = nn.Conv3d(in_channel, in_channel * 3, 1, bias=False)
        # self.out = nn.Conv3d(in_channel, in_channel, 1)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        # batch, channel, depth, height, width = input.shape
        batch, channel, height, width = input.shape  # torch.Size([1, 32, 32, 32])
        n_head = self.n_head  # 4

        norm = self.norm(input)  # torch.Size([1, 32, 32, 32])
        # print(norm.shape)
        qkv = self.qkv(norm)  # torch.Size([1, 96, 32, 32])
        # print(qkv.shape)
        # q, k, v = rearrange(qkv, 'b (qkv heads c) d h w -> qkv b heads c (d h w)', heads=n_head, qkv=3)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=n_head, qkv=3)  # torch.Size([3, 1, 4, 8, 1024])
        k = k.softmax(dim=-1)  # torch.Size([1, 4, 8, 1024])
        # pdb.set_trace()
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        # out = rearrange(out, 'b heads c (d h w) -> b (heads c) d h w', heads=n_head, h=height, w=width)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=n_head, h=height, w=width)
        out = self.out(out)
        return out + input


# class SelfAttention_fuse(nn.Module): 条件引导
#     def __init__(self, in_channel, n_head=4):
#         super().__init__()
#
#         self.n_head = n_head
#         self.norm = nn.GroupNorm(4, in_channel)
#         # self.out = nn.Conv3d(in_channel, in_channel, 1)
#         # self.defmgen=nn.Conv3d(in_channel,3,3,padding=1)
#         # self.nonlinear=nn.Conv3d(3, 3, 3, padding=1)
#         self.out = nn.Conv2d(in_channel, in_channel, 1)
#         self.defmgen=nn.Conv2d(in_channel,3,3,padding=1)
#         self.nonlinear=nn.Conv2d(3, 2, 3, padding=1)  # out_channels = 3
#         self.conv_v = nn.Conv2d(1, in_channel, 3, 1, 1)
#
#     def forward(self, q,k,v,size):
#         # batch, channel, depth, height, width = q.shape
#         batch, channel, height, width = q.shape
#
#         n_head = self.n_head
#         residual=q
#         norm_q = self.norm(q)
#         norm_k = self.norm(k)
#         norm_v = self.norm(self.conv_v(F.interpolate(v, size=q.shape[-2:], mode="bilinear")))
#
#         # pdb.set_trace()
#         qkv=torch.cat([norm_q,norm_k,norm_v],dim=1)
#         # q, k, v = rearrange(qkv, 'b (qkv heads c) d h w -> qkv b heads c (d h w)', heads=n_head, qkv=3)
#         q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=n_head, qkv=3)
#         k = k.softmax(dim=-1)
#         context = torch.einsum('bhdn,bhen->bhde', k, v)
#         out = torch.einsum('bhde,bhdn->bhen', context, q)
#         # out = rearrange(out, 'b heads c (d h w) -> b (heads c) d h w', heads=n_head, h=height, w=width)
#         out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=n_head, h=height, w=width)
#         out = self.out(out)
#         out=self.defmgen(out+residual)
#         # pdb.set_trace()
#         out=F.upsample_nearest(out,size)
#         out=self.nonlinear(out)
#         return out


class SelfAttention_fuse(nn.Module):
    def __init__(self, in_channel, n_head=4):
        super().__init__()

        self.nonlinear = nn.Conv2d(in_channel * 2, 2, 3, padding=1)

    def forward(self, q, k, size):
        # pdb.set_trace()
        out = F.upsample_nearest(torch.cat([q, k], dim=1), size)
        out = self.nonlinear(out)
        return out

class CondInjection(nn.Module):
    def __init__(self, fea_dim, cond_dim, hidden_dim, groups=32) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(8, hidden_dim, 3, padding=1, bias=False),  # add_n_channel * 2
            nn.GroupNorm(groups, hidden_dim),
            nn.SiLU(),
            # nn.Conv2d(hidden_dim * 4, hidden_dim * 2, 1, bias=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=True),
        )
        # self.x_conv = nn.Conv2d(fea_dim, hidden_dim, 1, bias=True)
        self.x_conv = nn.Conv2d(fea_dim + hidden_dim, hidden_dim, 1, bias=True)
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x, cond):  # torch.Size([2, 32, 256, 256])
        # pdb.set_trace()
        cond = self.body(cond)  # torch.Size([2, 64, 256, 256])
        # scale, shift = cond.chunk(2, dim=1)

        # x = self.x_conv(x)
        x = self.x_conv(torch.cat([x, cond], dim=1))
        # x = x * (1 + scale) + shift
        # x = x * (1 + cond)
        return x

class ResnetBlocWithAttn_reg(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0, with_attn=False, cond_dim=None):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock_reg(
            dim, dim_out, time_emb_dim, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention_reg(dim_out)
        self.cond_dim = cond_dim
        if cond_dim:  # 条件引导
            self.cond_inj = CondInjection(dim, dim, hidden_dim=dim, groups=1)  # 条件引导

    def forward(self, x, time_emb, cond):
        # pdb.set_trace()
        if self.cond_dim:  # 条件引导
            x = self.cond_inj(
                x, F.interpolate(cond, size=x.shape[-2:], mode="bilinear")
            )
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_time_emb=True,
        image_size=128,
        # opt=None
    ):
        super().__init__()

        if with_time_emb:
            time_dim = inner_channel
            self.time_mlp = nn.Sequential(
                TimeEmbedding_reg(inner_channel),  # 时间嵌入
                nn.Linear(inner_channel, inner_channel * 4),  # 4
                Swish(),  # 增加模型的非线性
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            time_dim = None
            self.time_mlp = None
        # self.opt=opt
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size[1]
        # downs = [nn.Conv3d(in_channel, inner_channel,kernel_size=3, padding=1)]
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn_reg(pre_channel, channel_mult, time_emb_dim=time_dim, dropout=dropout, with_attn=use_attn, cond_dim=True))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample_reg(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn_reg(pre_channel, pre_channel, time_emb_dim=time_dim,
                               dropout=dropout, with_attn=True, cond_dim=False),
            ResnetBlocWithAttn_reg(pre_channel, pre_channel,
                               time_emb_dim=time_dim, dropout=dropout, with_attn=False, cond_dim=False)
        ])


        ups_diff = []
        ups_regis = []
        ups_adapt = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            # print(use_attn)  False
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                feat_channel=feat_channels.pop()
                ups_diff.append(ResnetBlocWithAttn_reg(
                    pre_channel+feat_channel, channel_mult, time_emb_dim=time_dim, dropout=dropout, with_attn=use_attn, cond_dim=False))
                regischannel=pre_channel+feat_channel+channel_mult
                ups_regis.append(ResnetBlocWithAttn_reg(
                    regischannel, channel_mult, time_emb_dim=time_dim, dropout=dropout, with_attn=use_attn, cond_dim=False))
                ups_adapt.append(
                    SelfAttention_fuse(channel_mult)
                )
                pre_channel = channel_mult
            if not is_last:
                ups_adapt.append(nn.Identity())
                ups_diff.append(Upsample_reg(pre_channel))
                ups_regis.append(Upsample_reg(pre_channel))
                now_res = now_res*2

        self.ups_diff = nn.ModuleList(ups_diff)
        self.ups_regis = nn.ModuleList(ups_regis)
        self.ups_adapt = nn.ModuleList(ups_adapt)
        self.final_conv_reg = Block_reg(pre_channel, default(out_channel, in_channel))
        # self.final_attn=SelfAttention_fuse(1)
        self.final_conv_defm_reg = Block_reg(pre_channel+2,2,groups=3)  # dim_out = 3  # 原始 pre_channel+4 groups=4
        # self.final_conv_defm_reg = Block_reg(pre_channel+1,2,groups=1)  # dim_out = 3

    def forward(self, x, x_m, time):  # torch.Size([2, 3, 256, 256]) torch.Size([2, 1, 256, 256]) torch.Size([2])
        # pdb.set_trace()
        # input_size=(x.size(2),x.size(3),x.size(4))
        input_size = (x.size(2), x.size(3))  # (256, 256)
        t = self.time_mlp(time) if exists(self.time_mlp) else None  # torch.Size([2, 8])
        feats = []
        for layer in self.downs:
            # pdb.set_trace()
            if isinstance(layer, ResnetBlocWithAttn_reg):
                # print('down, attention')
                # x = layer(x, t)
                x = layer(x, t, x_m)
            else:
                # print('down')
                x = layer(x)  # False
            feats.append(x)
        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn_reg):
                # print('mid, attention')
                # x = layer(x, t)
                x = layer(x, t, x_m)
            else:
                # print('mid')
                x = layer(x)
        x_1=x
        x_2=x
        defm=[]
        # x_1vis=[]
        # pdb.set_trace()
        for layerd,layerr,layera in zip(self.ups_diff, self.ups_regis, self.ups_adapt):
            if isinstance(layerd, ResnetBlocWithAttn_reg):
                # print('up, attention')
                feat=feats.pop()
                x_1 = layerd(torch.cat((x_1, feat), dim=1), t, x_m)  # 原始
                x_2 = layerr(torch.cat((x_2, feat,x_1), dim=1), t, x_m)  # 原始
                # defm_=layera(x_2,x_1,x_1,input_size)  # 原始
                # defm_=layera(x_2,x_1,x_m,input_size)  # 条件引导
                defm_ = layera(x_2, x_1, input_size)
                defm.append(defm_)  # 8 * torch.Size([1, 3, 256, 256])  # 原始
            else:
                # print('up')
                x_1 = layerd(x_1)
                x_2 = layerr(x_2)  # 原始
        # pdb.set_trace()
        recon=self.final_conv_reg(x_1)  # torch.Size([2, 1, 256, 256])  # 原始
        # defm = self.final_conv_defm_reg(recon)  # torch.Size([2, 1, 256, 256])
        defm=torch.stack(defm,dim=1)  # torch.Size([2, 8, 2, 256, 256])  # 原始
        atest = self.final_conv_defm_reg(torch.cat((x_2, recon), dim=1)).unsqueeze_(1)  # torch.Size([2, 1, 2, 256, 256])  # 原始
        defm=torch.cat([defm,atest],dim=1)  # torch.Size([2, 9, 2, 256, 256])  # 原始
        defm=torch.mean(defm,dim=1)  # torch.Size([2, 2, 256, 256])  # 原始
        # # defm = torch.cat([recon, recon], dim=1)  # 原始
        return recon, defm  # torch.Size([1, 1, 256, 256]), torch.Size([1, 2, 256, 256]) 重建图像和额外的 变形 特征图

class Dense3DSpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """
    def __init__(self, volsize=(256, 256), mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler

            ：param size：空间转换器块的输入大小
            ：param mode：网格采样器的插值方法
        """
        super(Dense3DSpatialTransformer, self).__init__()

        # Create sampling grid
        size = volsize  # (256, 256)
        vectors = [ torch.arange(0, s) for s in size]  # [torch.Size([256]), torch.Size([256])]
        grids = torch.meshgrid(vectors)  # [torch.Size([256, 256]), torch.Size([256, 256])]
        grid = torch.stack(grids) # torch.Size([2, 224, 224])
        grid = torch.unsqueeze(grid, 0)  # torch.Size([1, 2, 224, 224])
        grid = grid.type(torch.FloatTensor).cuda()
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow): # src: torch.Size([1, 1, 512, 512])
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        推送src并flow经空间转换块
            ：param src：原始运动图像
            ：param flow：U-Net的输出
        """

        # print(src.shape, flow.shape, self.grid.shape) #torch.Size([16, 1, 224, 224]) torch.Size([16, 2, 224, 224]) torch.Size([1, 2, 224, 224])
        new_locs = self.grid + flow # torch.Size([1, 2, 224, 224])
        shape = flow.shape[2:] # torch.Size([224, 224])


        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...].clone() / (shape[i] - 1) - 0.5)


        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1) # torch.Size([16, 224, 224, 2])
            new_locs = new_locs[..., [1,0]] # torch.Size([16, 224, 224, 2])
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2,1,0]]

        return F.grid_sample(src, new_locs, mode=self.mode, padding_mode='border', align_corners=True)  # , new_locs