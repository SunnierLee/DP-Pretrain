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
bins = 2
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

save_dir = "./mode_{}_n{}_s{}_ds{}_q{}_b{}".format(data_name, sample_num, sigma, ds, q, bins)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
os.mkdir(os.path.join(save_dir, 'clean'))
os.mkdir(os.path.join(save_dir, 'noisy'))
os.mkdir(os.path.join(save_dir, 'lr_noisy'))

c = 0
for _ in range(10000):
    for x, y in dataloader:
        # x = F.interpolate(x, size=(32, 32), mode=mode)
        # x = x.repeat((1, 3, 1, 1))
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

            sensitivity = np.sqrt((ds**2) * np.prod(x_cls.shape[1:]))

            print(sigma, sensitivity)

            x_cls = F.interpolate(x_cls, scale_factor=ds)
            x_cls_shape = x_cls.shape[-3:]
            x_cls = x_cls.view(x_cls.shape[0], -1)
            x_majority = []
            x_majority_noisy = []
            for i in range(x_cls.shape[1]):
                hist = torch.histc(x_cls[:, i], bins=bins, min=0, max=1)
                x_majority.append(torch.argmax(hist))
                hist = hist + torch.randn_like(hist) * sensitivity * sigma
                x_majority_noisy.append(torch.argmax(hist))
            # print(x_majority_noisy)
            x_majority = torch.tensor(x_majority).view(1, *x_cls_shape).float() / (bins-1)
            x_majority_noisy = torch.tensor(x_majority_noisy).view(1, *x_cls_shape).float() / (bins-1)

            x_majority_noisy_lr = (x_majority_noisy*255).cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)[0]
            if x_majority_noisy_lr.shape[-1] == 1:
                x_majority_noisy_lr = x_majority_noisy_lr[..., 0]
            Image.fromarray(x_majority_noisy_lr).save(os.path.join(out_dir_lr_noisy, '{}.png'.format(c)))

            x_majority_noisy = F.interpolate(x_majority_noisy.float(), size=x.shape[-2:], mode=mode)

            x_majority_noisy = (x_majority_noisy.cpu()*255).permute(0, 2, 3, 1).numpy().astype(np.uint8)[0]
            if x_majority_noisy.shape[-1] == 1:
                x_majority_noisy = x_majority_noisy[..., 0]
            Image.fromarray(x_majority_noisy).save(os.path.join(out_dir_noisy, '{}.png'.format(c)))

            x_majority = (x_majority.cpu()*255).permute(0, 2, 3, 1).numpy().astype(np.uint8)[0]
            if x_majority.shape[-1] == 1:
                x_majority = x_majority[..., 0]
            Image.fromarray(x_majority).save(os.path.join(out_dir_clean, '{}.png'.format(c)))
        c += 1
        if c == sample_num:
            break
    if c == sample_num:
        break