import torchvision
import torch
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

from stylegan3.dataset import ImageFolderDataset


data_name = "mnist"
attr = "Male"
use_labels = True
sample_num = 5
sigma = 5
ds = 0.5
batch_size = 6000
mode = "bilinear"

if "celeba" == data_name:
    num_classes = 2
    data_dir = ''
    dataset = ImageFolderDataset(data_dir, split='train', resolution=32, use_labels=use_labels, attr=attr)
elif "camelyon" == data_name:
    num_classes = 2
    data_dir = ''
    dataset = ImageFolderDataset(data_dir, split='train', resolution=32, use_labels=use_labels, attr=attr)
elif data_name == "mnist":
    num_classes = 10
    data_dir = ''
    dataset = torchvision.datasets.MNIST(data_dir, download=True, train=True, transform=transforms.ToTensor())
elif data_name == "fmnist":
    num_classes = 10
    data_dir = ''
    dataset = torchvision.datasets.FashionMNIST(data_dir, download=True, train=True, transform=transforms.ToTensor())
print("number of sensitive data: ", len(dataset))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
q = batch_size / len(dataset)

save_dir = "./mean_{}_{}_{}_{}".format(data_name, sample_num, sigma, q)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
os.mkdir(os.path.join(save_dir, 'clean'))
os.mkdir(os.path.join(save_dir, 'noisy'))
os.mkdir(os.path.join(save_dir, 'lr_noisy'))

c = 0
for _ in range(10000):
    for x, y in dataloader:
        if x.max() > 2:
            x = x/255.
            if num_classes > 1:
                y = torch.argmax(y, dim=1)
        for cls in range(num_classes):
            out_dir_clean = os.path.join(save_dir, 'clean', str(cls).zfill(6))
            out_dir_noisy = os.path.join(save_dir, 'noisy', str(cls).zfill(6))
            out_dir_lr_noisy = os.path.join(save_dir, 'lr_noisy', str(cls).zfill(6))
            if not os.path.exists(out_dir_clean):
                os.mkdir(out_dir_clean)
            if not os.path.exists(out_dir_noisy):
                os.mkdir(out_dir_noisy)
            if not os.path.exists(out_dir_lr_noisy):
                os.mkdir(out_dir_lr_noisy)
            
            x_cls = x if num_classes==1 else x[y==cls]

            x_norm = torch.sum(x_cls**2, dim=[1, 2, 3], keepdim=True).sqrt()
            C = x_norm.max()

            sensitivity_m = np.sqrt((ds**2) * np.prod(x_cls.shape[1:])) / x_cls.shape[0]
            # sensitivity_m = C / x_cls.shape[0]
            # sensivity_std = np.sqrt(1 / np.prod(x_cls.shape[1:])) / x_cls.shape[0] / num_classes

            xm = torch.mean(x_cls, dim=0, keepdim=True)
            # xstd = torch.std(x_cls, dim=0, keepdim=True).mean()

            # xstd_noisy = xstd + np.random.randn() * sensitivity_m * sigma
            # sigma_m = xstd_noisy / sensitivity_m / gamma
            sigma_m = sigma
            print(C, sigma_m, sensitivity_m)
            xm = F.interpolate(xm, scale_factor=ds)
            xm = xm + torch.randn_like(xm) * sensitivity_m * sigma_m
            xm = xm.clamp(0., 1.)
            xm_lr = (xm.cpu().permute(0, 2, 3, 1) * 255.).numpy().astype(np.uint8)[0]
            if xm_lr.shape[-1] == 1:
                xm_lr = xm_lr[..., 0]
            Image.fromarray(xm_lr).save(os.path.join(out_dir_lr_noisy, '{}.png'.format(c)))
            xm = F.interpolate(xm, size=x.shape[-2:], mode=mode)

            xm = xm.cpu().permute(0, 2, 3, 1) * 255.
            xm = xm.numpy().astype(np.uint8)[0]
            if xm.shape[-1] == 1:
                xm = xm[..., 0]
            Image.fromarray(xm).save(os.path.join(out_dir_noisy, '{}.png'.format(c)))
            xm = torch.mean(x_cls, dim=0, keepdim=True)
            xm = xm.clamp(0., 1.)
            xm = F.interpolate(xm, size=x.shape[-2:], mode=mode)
            xm = xm.cpu().permute(0, 2, 3, 1) * 255.
            xm = xm.numpy().astype(np.uint8)[0]
            if xm.shape[-1] == 1:
                xm = xm[..., 0]
            Image.fromarray(xm).save(os.path.join(out_dir_clean, '{}.png'.format(c))) 
        c += 1
        if c == sample_num:
            break
    if c == sample_num:
        break
# if c < sample_num:
#     for x, y in dataloader:
#         if 'zip' in data_dir:
#             x = x/255.
#         for cls in range(num_classes):
#             out_dir = os.path.join(save_dir, str(cls).zfill(6))
#             if not os.path.exists(out_dir):
#                 os.mkdir(out_dir)
#             x_cls = x if num_classes==1 else x[y==cls]
#             x_cls = F.interpolate(x_cls, scale_factor=ds)
#             x_norm = torch.sum(x_cls**2, dim=[1, 2, 3], keepdim=True).sqrt()
#             C = x_norm.max()
#             sensitivity = C * sigma / x_norm.shape[0]
#             xm = torch.mean(x_cls, dim=0, keepdim=True)
#             xstd = torch.std(x_cls, dim=0, keepdim=True).mean()
#             print(C, xstd)
#             xm = xm + torch.randn_like(xm) * sensitivity * sigma
#             xm = xm.clamp(0., 1.)
#             xm = F.interpolate(xm, size=x.shape[-2:], mode=mode)
#             xm = xm.cpu().permute(0, 2, 3, 1) * 255.
#             xm = xm.numpy().astype(np.uint8)[0]
#             if xm.shape[-1] == 1:
#                 xm = xm[..., 0]
#             Image.fromarray(xm).save(os.path.join(out_dir, '{}.png'.format(c)))
#             xm = torch.mean(x_cls, dim=0, keepdim=True)
#             xm = xm.clamp(0., 1.)
#             xm = F.interpolate(xm, size=x.shape[-2:], mode=mode)
#             xm = xm.cpu().permute(0, 2, 3, 1) * 255.
#             xm = xm.numpy().astype(np.uint8)[0]
#             if xm.shape[-1] == 1:
#                 xm = xm[..., 0]
#             Image.fromarray(xm).save(os.path.join(out_dir, '{}_clean.png'.format(c)))
#         c += 1
#         break