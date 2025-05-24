import pdb

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class gradientLoss(nn.Module):
    def __init__(self, penalty='l1'):
        super(gradientLoss, self).__init__()
        self.penalty = penalty

    def forward(self, input):
        # dD = torch.abs(input[:, :, 1:, :, :] - input[:, :, :-1, :, :])
        # dH = torch.abs(input[:, :, :, 1:, :] - input[:, :, :, :-1, :])
        # dW = torch.abs(input[:, :, :, :, 1:] - input[:, :, :, :, :-1])
        dD = torch.abs(input[:, 1:, :, :] - input[:, :-1, :, :])
        dH = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])
        dW = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])
        if(self.penalty == "l2"):
            dD = dD * dD
            dH = dH * dH
            dW = dW * dW
        loss = (torch.mean(dD) + torch.mean(dH) + torch.mean(dW)) / 3.0
        return loss

class calculate_moment(nn.Module):  # order = 0.5
    def __init__(self, order):
        super(calculate_moment, self).__init__()
        self.order = order
        self.loss = nn.L1Loss()

    def forward(self, image1, image2):
        moment1 = self.get_moments(image1, order=self.order)
        moment2 = self.get_moments(image2, order=self.order)
        loss = self.loss(moment1, moment2)
        return loss

    def get_moments(self, image_ori, order):
        # if order<0 or order>=1:
        #     raise ValueError("阶数必须是一个大于等于0小于1的正分数。")
        out = 0
        for m in range(len(image_ori)):
            image = image_ori[m]
            # fenzi, fenmu = map(int, torch.tensor(np.absolute(np.log(order))))
            # n = image.size(0) * image.size(1) * image.size(2)
            junzhi = image.mean()
            variance = image.var()
            std_dev = variance.sqrt()
            moment = 0

            # pdb.set_trace()
            for i in range(image.size(1)):
                # pdb.set_trace()
                for j in range(image.size(2)):
                    diff = image[m, i, j] - junzhi
                    moment += (diff/std_dev)**order
            pdb.set_trace()
            moment /= (image.size(0) * image.size(1) * image.size(2))**order
            out += moment
        return out

#######################分数阶微分################################################################################
# 定义一个简单的卷积层
class GaussianFilter(nn.Module):
    def __init__(self, size, sigma):
        super(GaussianFilter, self).__init__()
        self.size = size
        self.sigma = sigma
        self.filter = torch.zeros((size, size))
        x = torch.linspace(-size//2, size//2, size)
        y = x.unsqueeze(1).repeat(4, size)  # 1 4
        g = torch.exp(-((x**2 + y**2) / (2 * sigma**2)))
        g /= torch.sum(g)
        # self.filter = g.view(1, 1, size, size).cuda()
        self.filter = g.view(1, 4, size, size).cuda()

    def forward(self, img):
        img_pad = F.pad(img, (self.size//2, self.size//2, self.size//2, self.size//2), mode='reflect')
        # pdb.set_trace()
        filtered_img = F.conv2d(img_pad, self.filter, groups=1)
        return filtered_img

# 定义分数阶导数层
class FractionalDerivative(nn.Module):
    def __init__(self, alpha):
        super(FractionalDerivative, self).__init__()
        self.alpha = alpha
        self.beta = 1 - alpha
        self.n = int(torch.ceil(torch.log(torch.tensor(self.beta)) / torch.log(torch.tensor(1 - self.alpha))))
        self.filter = GaussianFilter(2 * self.n + 1, self.n / 2)

    def forward(self, img):
        # pdb.set_trace()
        frac_derivative = self.filter(img)
        return frac_derivative

# 计算两个图像之间的分数阶损失
def fractional_loss(img1, img2, alpha):
    # pdb.set_trace()
    frac_derivative1 = FractionalDerivative(alpha)(img1)
    frac_derivative2 = FractionalDerivative(alpha)(img2)
    loss = torch.norm(frac_derivative1 - frac_derivative2)
    return loss

#################################################################################################################
class SobelFilter(nn.Module):
    def __init__(self):
        super(SobelFilter, self).__init__()
        # 定义水平和垂直方向的Sobel算子
        # self.filter_x = torch.tensor([
        #     [-1, 0, 1],
        #     [-2, 0, 2],
        #     [-1, 0, 1]
        # ], dtype=torch.float32).cuda()
        # 拉普拉斯算子
        self.filter_x = torch.tensor([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ], dtype=torch.float32).cuda()
        self.filter_y = self.filter_x.t()
        self.filter_x = self.filter_x.unsqueeze(0).unsqueeze(0)
        self.filter_y = self.filter_y.unsqueeze(0).unsqueeze(0)

    def forward(self, img):
        # 使用F.conv2d进行卷积
        img = img.view(1, 1, 4, 256, 256).mean(dim=2)
        img_pad = F.pad(img, (1, 1, 1, 1), mode='reflect')
        grad_x = F.conv2d(img_pad, self.filter_x, groups=1)
        grad_y = F.conv2d(img_pad, self.filter_y, groups=1)
        return grad_x, grad_y

def Sobel_loss(img1, img2):
    # 计算两个图像的水平和垂直梯度
    sobel_filter = SobelFilter()
    grad_x1, grad_y1 = sobel_filter(img1)
    grad_x2, grad_y2 = sobel_filter(img2)
    # 计算梯度的L2损失
    loss = (torch.norm(grad_x1 - grad_x2) + torch.norm(grad_y1 - grad_y2)) / 2
    return loss

class crossCorrelation3D(nn.Module):
    def __init__(self, in_ch, kernel=(9, 9), gamma=1):  # kernel=(9, 9, 9)
        super(crossCorrelation3D, self).__init__()
        self.in_ch = in_ch
        self.kernel = kernel
        self.gamma=gamma
        # self.filt = (torch.ones([1, in_ch, self.kernel[0], self.kernel[1], self.kernel[2]])).to('cuda:0')
        self.filt = (torch.ones([1, 1, self.kernel[0], self.kernel[1]])).to('cuda:0')


    def forward(self, input, target,flow):
        min_max = (-1, 1)
        target = (target - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
        
        II = input * input
        TT = target * target
        IT = input * target

        flow=F.sigmoid(flow)**self.gamma
        pad = (int((self.kernel[0] - 1) / 2))
        T_sum = F.conv2d(target, self.filt, stride=1, padding=pad)
        I_sum = F.conv2d(input, self.filt, stride=1, padding=pad)#*flow
        TT_sum = F.conv2d(TT, self.filt, stride=1, padding=pad)
        II_sum = F.conv2d(II, self.filt, stride=1, padding=pad)#*flow
        IT_sum = F.conv2d(IT, self.filt, stride=1, padding=pad)
        kernelSize = self.kernel[0] * self.kernel[1]  # * self.kernel[2]
        Ihat = I_sum / kernelSize
        That = T_sum / kernelSize

        cross = IT_sum - Ihat*T_sum - That*I_sum + That*Ihat*kernelSize
        T_var = TT_sum - 2*That*T_sum + That*That*kernelSize
        I_var = II_sum - 2*Ihat*I_sum + Ihat*Ihat*kernelSize
        cc = cross*cross*flow / (T_var*I_var+1e-5)

        loss = -1.0 * torch.mean(cc)
        return loss


# 均方误差（MSE）
def mse_loss(x, y):
    return F.mse_loss(x, y)


def normalized_cross_correlation(x, y):
    """
    Calculate the Normalized Cross-Correlation (NCC) between two tensors,
    matching MATLAB's `corrcoef` behavior.

    Args:
        x (torch.Tensor): The first tensor with dimensions [B, C, H, W].
        y (torch.Tensor): The second tensor with dimensions [B, C, H, W].

    Returns:
        torch.Tensor: The NCC tensor with dimensions [B].
    """
    assert x.shape == y.shape, "The shapes of x and y must be the same."
    # pdb.set_trace()
    """
    完全复现 MATLAB 代码行为的 NCC 计算：
    1. 输入为 [H, W] 或 [1, 1, H, W] 的张量
    2. 与 MATLAB 的 std 计算（总体标准差）保持一致
    3. 严格复现 mean(mean(corrcoef(...))) 操作
    """
    # 转换为 numpy 数组并去除冗余维度
    grayImg1 = np.squeeze(x.cpu().numpy()).astype(np.float64)
    grayImg2 = np.squeeze(y.cpu().numpy()).astype(np.float64)

    # 计算均值
    mean1 = np.mean(grayImg1)
    mean2 = np.mean(grayImg2)

    # 计算标准差（MATLAB 风格，ddof=0 表示总体标准差）
    std1 = np.std(grayImg1, ddof=0)
    std2 = np.std(grayImg2, ddof=0)

    # 归一化（避免除以零）
    eps = 1e-8
    grayImg1_norm = (grayImg1 - mean1) / (std1 + eps)
    grayImg2_norm = (grayImg2 - mean2) / (std2 + eps)

    # 计算相关系数矩阵
    corr_matrix = np.corrcoef(grayImg1_norm.ravel(), grayImg2_norm.ravel())

    # 严格复现 MATLAB 的 mean(mean(...)) 操作
    # pdb.set_trace()
    # ncc = np.mean(corr_matrix)
    ncc = corr_matrix[0, 1]

    return ncc

def mutual_information(x, y):
    """
    Calculate the Mutual Information between two tensors.

    Args:
        x (torch.Tensor): The first tensor with dimensions [B, 1, H, W].
        y (torch.Tensor): The second tensor with dimensions [B, 1, H, W].

    Returns:
        torch.Tensor: The Mutual Information tensor with dimensions [B].
    """
    # Ensure the tensors have the same shape
    assert x.shape == y.shape, "The shapes of x and y must be the same."

    # Flatten the tensors
    x = x.flatten(1, 2)
    y = y.flatten(1, 2)

    # Calculate the entropy of x and y
    x_entropy = -torch.mean(torch.sum(x * torch.log(x + 1e-8), dim=1))
    y_entropy = -torch.mean(torch.sum(y * torch.log(y + 1e-8), dim=1))

    # Calculate the joint entropy
    xy = torch.cat((x, y), dim=1)
    xy_entropy = -torch.mean(torch.sum(xy * torch.log(xy + 1e-8), dim=1))

    # Calculate the mutual information
    mi = x_entropy + y_entropy - xy_entropy

    return mi

def fractional_derivative(image_tensor, order, alpha, scale):
    """
    计算一批图像张量的分数阶导数
    :param image_tensor: 输入图像张量，形状为[b, 4, M, N]，b为批量大小，4为通道数，M、N为图像高和宽
    :param order: 分数阶数
    :param alpha: 分数阶参数
    :param scale: 尺度参数
    :return: 计算分数阶导数后的图像张量，形状与输入相同
    """
    batch_size, num_channels, M, N = image_tensor.shape
    result_tensor = np.zeros_like(image_tensor)
    for batch_idx in range(batch_size):
        for channel_idx in range(num_channels):
            image = image_tensor[batch_idx, channel_idx]  # 获取当前批次、当前通道的图像
            kernel_size = int(2 * scale + 1)
            kernel = np.zeros((kernel_size, kernel_size))
            center = int(kernel_size / 2)

            for i in range(kernel_size):
                for j in range(kernel_size):
                    x = i - center
                    y = j - center
                    kernel[i, j] = ((x ** 2 + y ** 2) ** alpha) * (np.math.log(x ** 2 + y ** 2 + np.finfo(float).eps) ** (order - alpha))

            kernel /= np.sum(kernel)  # 归一化
            result = cv2.filter2D(image, -1, kernel)
            result_tensor[batch_idx, channel_idx] = result
    return result_tensor

def fractional_loss_2(img1, img2, alpha):
    # pdb.set_trace()
    frac_derivative1 = fractional_derivative(img1, alpha, 1.0)
    frac_derivative2 = fractional_derivative(img2, alpha, 1.0)
    loss = torch.norm(frac_derivative1 - frac_derivative2)
    return loss

class FractionalDerivativeLoss(nn.Module):
    def __init__(self, alpha=0.5, kernel_size=30):
        super(FractionalDerivativeLoss, self).__init__()
        self.alpha = alpha
        self.kernel_size = kernel_size

        # 将 alpha 转换为张量以支持自动设备同步
        self.register_buffer("alpha_tensor", torch.tensor(alpha, dtype=torch.float32))

        # 生成 Grünwald-Letnikov 卷积核
        self.register_buffer("kernel_i", self._create_gl_kernel(direction="i"))
        self.register_buffer("kernel_j", self._create_gl_kernel(direction="j"))

    def _create_gl_kernel(self, direction):
        # 动态获取设备信息
        device = self.alpha_tensor.device

        # 生成索引 k 并确保在相同设备上
        k = torch.arange(self.kernel_size, dtype=torch.float32, device=device)

        # 计算广义二项式系数（分子和分母分开计算）
        numerator = torch.exp(torch.lgamma(self.alpha_tensor + 1))  # Γ(α+1)
        denominator = torch.exp(
            torch.lgamma(k + 1) +  # Γ(k+1)
            torch.lgamma(self.alpha_tensor - k + 1)  # Γ(α -k +1)
        )

        # 计算系数并添加数值稳定性项
        coeff = (-1.0) ** k * numerator / (denominator + 1e-8)  # (-1)^k * Γ(α+1) / [Γ(k+1)Γ(α−k+1)]

        # 调整方向
        if direction == "i":
            return coeff.view(1, 1, 1, -1)#.repeat(4,1,1,1)  # 行方向卷积核 (1, kernel_size)
        else:
            return coeff.view(1, 1, -1, 1)#.repeat(4,1,1,1)  # 列方向卷积核 (kernel_size, 1)

    def _apply_derivative(self, x, kernel, direction):
        # pdb.set_trace()
        # 根据方向确定填充方式
        if direction == "i":
            padding = (kernel.shape[-1] - 1, 0, 0, 0)  # 左侧填充
        else:
            padding = (0, 0, kernel.shape[-2] - 1, 0)  # 顶部填充

        x_pad = F.pad(x, padding, mode='constant', value=0)
        return F.conv2d(x_pad, kernel, stride=1, padding=0, groups=x.size(1))

    def forward(self, X_hat_res, Z_hat_res):
        # 计算 X 的分数阶导数
        if X_hat_res.size(1) != 1:
            X_hat_res = X_hat_res[:, :1, :, :]  # .mean(dim=1, keepdim=True)
            Z_hat_res = Z_hat_res[:, :1, :, :]  # .mean(dim=1, keepdim=True)

        Di_X = self._apply_derivative(X_hat_res, self.kernel_i, "i")
        Dj_X = self._apply_derivative(X_hat_res, self.kernel_j, "j")
        term_X = torch.abs(Di_X * Dj_X).sum()

        # 计算 Z 的分数阶导数
        Di_Z = self._apply_derivative(Z_hat_res, self.kernel_i, "i")
        Dj_Z = self._apply_derivative(Z_hat_res, self.kernel_j, "j")
        term_Z = torch.abs(Di_Z * Dj_Z).sum()

        return term_Z - term_X