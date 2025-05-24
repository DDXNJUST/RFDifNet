import argparse
import pathlib
import pdb
import warnings

import os
import cv2
import numpy as np
import kornia
import torch.backends.cudnn
import torch.cuda
import torch.utils.data
import torchvision
from torch import Tensor
from tqdm import tqdm

from functions.affine_transform import AffineTransform
from functions.elastic_transform import ElasticTransform

from skimage import io as skio
from pathlib import Path
import scipy

def flow2rgb(flow_map: [Tensor], max_value: None):
    # pdb.set_trace()
    flow_map_np = flow_map.squeeze().detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    rgb_map = np.ones((3, h, w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    rgb_flow = rgb_map.clip(0, 1)
    # pdb.set_trace()
    return rgb_flow
def save_flow(flow: [Tensor], dst: pathlib.Path, im_name: str = ''):
    rgb_flow = flow2rgb(flow, max_value=None)  # (3, 512, 512) type; numpy.ndarray
    im_s = rgb_flow if type(rgb_flow) == list else [rgb_flow]
    im_cv = (im_s[0] * 255).astype(np.uint8).transpose(1, 2, 0)
    # pdb.set_trace()
    cv2.imwrite(str(dst)+'/'+im_name, im_cv)

def save_flow_2(flow: [Tensor], dst: pathlib.Path, im_name: str = '', step=16):
    # pdb.set_trace()
    flow = flow[-1, :, :, :].permute(1, 2, 0) * 255.
    flow = flow.detach().cpu().numpy()
    image = np.full((256, 256, 3), 255.0)  # image = np.ones([256, 256, 3])
    h, w = image.shape[:2]  # h w 3
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T  # h w 2

    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(image, (x1, y1), (x2, y2), (0, 0, 0), 1, tipLength=0.2)
    cv2.imwrite(str(dst)+'/'+im_name, image)

    # pdb.set_trace()


class getDeformableImages:
    """
    principle: ir -> ir_warp
    """
    def __init__(self):
        # hardware settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # deformable transforms
        self.elastic = ElasticTransform(kernel_size=101, sigma=16)
        self.affine  = AffineTransform(translate=0.01)

    @torch.no_grad()
    def __call__(self, ir_folder: pathlib.Path, vi_folder: pathlib.Path, dst: pathlib.Path):

        # get images list
        ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp', '.tif', '.mat']]
        # vi_list = [x for x in sorted(vi_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp', '.tif']]

        # starting generate deformable infrared image 开始生成可变形红外图像
        loader = tqdm(zip(ir_list))
        index = 1
        for ir_path in loader:
            print(index, str(ir_path[0].name))  # ir_path.name
            index += 1
            name = ir_path[0].name  # str(ir_path[0])  #
            loader.set_description(f'warp: {name}')
            name_disp = name.split('.')[0] + '_disp.npy'

            # read images
            ir = self.imread(ir_path, unsqueeze=True).to(self.device)  # torch.Size([1, 1, 256, 256])
            # vi = self.imread(vi_path, unsqueeze=True).to(self.device)  # torch.Size([1, 4, 64, 64])

            # get deformable images 获取可变形图像
            ir_affine, affine_disp = self.affine(ir)  # torch.Size([1, 1, 256, 256]) torch.Size([1, 256, 256, 2])
            ir_elastic, elastic_disp = self.elastic(ir_affine)  # torch.Size([1, 1, 256, 256]) torch.Size([1, 256, 256, 2])
            disp = affine_disp + elastic_disp  # cumulative disp grid [batch_size, height, weight, 2]
            ir_warp = ir_elastic

            _, _, h, w = ir_warp.shape  # torch.Size([1, 1, 256, 256])
            grid = kornia.utils.create_meshgrid(h, w, device=ir_warp.device).to(ir_warp.dtype)  # torch.Size([1, 256, 256, 2])
            grid = grid.permute(0, 3, 1, 2)  # torch.Size([1, 2, 256, 256])
            disp = disp.permute(0, 3, 1, 2)
            new_grid = grid + disp  # torch.Size([1, 2, 256, 256])
            # draw grid
            img_grid = self._draw_grid(ir.squeeze().cpu().numpy(), 24)
            #
            new_grid = new_grid.permute(0, 2, 3, 1)
            warp_grid = torch.nn.functional.grid_sample(img_grid.unsqueeze(0), new_grid, padding_mode='border', align_corners=False)
            # raw image w/o warp 未扭曲的原始图像
            # ir_raw_grid  = 0.8 * ir + 0.2 * img_grid
            ir_raw_grid  = 10.0 * ir + 0.2 * img_grid
            ir_raw_grid  = torch.clamp(ir_raw_grid, 0, 1)
            # warped grid & warped ir image 扭曲的栅格和扭曲的红外图像
            # ir_warp_grid = 0.8 * ir_warp + 0.2 * warp_grid
            ir_warp_grid = 10.0 * ir_warp + 0.2 * warp_grid
            ir_warp_grid = torch.clamp(ir_warp_grid, 0, 1)
            # disp
            disp_npy = disp.data.cpu().numpy()

            # save disp
            if not os.path.exists(dst):
                os.makedirs(dst)
            print('dst', dst)
            np.save(dst / name_disp, disp_npy)
            # save deformable images
            # self.imsave(vi, dst / 'vi_gray', name)
            self.imsave(ir_warp, dst / 'ir_warp', name)
            self.imsave(warp_grid,  dst / 'warp_grid', name)  #
            self.imsave(ir_warp_grid, dst / 'ir_warp_grid', name)
            self.imsave(ir_raw_grid, dst / 'ir_raw_grid', name)
            save_flow(-disp, dst / 'disp', name[:-4]+'.jpg')  #Path('../Simulated/Geo/0.01 51 12/disp.jpg'))


    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE, unsqueeze=False):  #
        # im_cv = cv2.imread(str(path), flags)
        im_cv = skio.imread(str(path))  #scipy.io.loadmat(path)['pan']  #  (str(path[0]))  # str(path)  ['rgbn'] path
        # im_cv = np.mean(im_cv, axis=2)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        # im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        im_ts = kornia.utils.image_to_tensor(im_cv / 2047.).type(torch.FloatTensor)  # 2047 5000. 8000
        return im_ts.unsqueeze(0) if unsqueeze else im_ts

    @staticmethod
    def imsave(im_s: [Tensor], dst: pathlib.Path, im_name: str = ''):
        """
        save images to path
        :param im_s: image(s)
        :param dst: if one image: path; if multiple images: folder path
        :param im_name: name of image
        """
        im_s = im_s if type(im_s) == list else [im_s]
        dst = [dst / str(i + 1).zfill(3) / im_name for i in range(len(im_s))] if len(im_s) != 1 else [dst / im_name]
        for im_ts, p in zip(im_s, dst):
            im_ts = im_ts.squeeze().cpu()
            p.parent.mkdir(parents=True, exist_ok=True)
            # im_cv = kornia.utils.tensor_to_image(im_ts) * 255.
            im_cv = kornia.utils.tensor_to_image(im_ts) * 8000.  # 8000.  #5000.  # 2047.  #
            # print(p)
            # cv2.imwrite(str(p), im_cv)
            print('str(p)', str(p))
            skio.imsave(str(p), im_cv.astype(np.uint16))
        # im_ts = im_s[0].squeeze().cpu()
        # im_cv = kornia.utils.tensor_to_image(im_ts) * 2047.
        # skio.imsave(str(dst), im_cv.astype(np.uint16))

    @staticmethod
    def _draw_grid(im_cv, grid_size: int = 10):
        # im_cv = np.mean(im_cv, axis=0)
        # im_gd_cv = np.full_like(im_cv, 255.0)
        im_gd_cv = np.full_like(im_cv, 2047.0)  # (256, 256) 2047.0 8000 5000 im_cv[0,:,:]
        # im_gd_cv = cv2.cvtColor(im_gd_cv, cv2.COLOR_GRAY2BGR)  # 灰度转化成BGR (256, 256, 3)
        np.tile(np.expand_dims(im_gd_cv, -1), (1, 1, 4))

        # color = (0, 0, 255)
        color = (0, 0, 2047)  # 2047 8000 5000

        height, width = im_cv.shape  # 256 256
        for x in range(0, width - 1, grid_size):
            cv2.line(im_gd_cv, (x, 0), (x, height), color, 1, 1)
        for y in range(0, height - 1, grid_size):
            cv2.line(im_gd_cv, (0, y), (width, y), color, 1, 1)

        # im_gd_ts = kornia.utils.image_to_tensor(im_gd_cv / 255.).type(torch.FloatTensor).cuda()
        im_gd_ts = kornia.utils.image_to_tensor(im_gd_cv / 2047.).type(torch.FloatTensor).cuda()  # 2047. 8000 5000
        return im_gd_ts


def hyper_args():
    """
    get hyper parameters from args
    """

    parser = argparse.ArgumentParser(description='Generating deformable testing data')  # 生成可变形测试数据
    parser.add_argument('--ir', default='', type=pathlib.Path)  #  /home/dwx/dwx/datasets/Pavia/pan
    parser.add_argument('--vi', default='', type=pathlib.Path)
    parser.add_argument('--dst', default='', help='fuse image save folder', type=pathlib.Path)  #  /home/dwx/dwx/datasets/Pavia/pan_warp/

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = hyper_args()
    data = getDeformableImages()
    data(ir_folder=args.ir, vi_folder=args.vi, dst=args.dst)
