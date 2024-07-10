import torch
import torch as ch
from torch import nn
import torch.nn.functional as F


# Model (from KakaoBrain: https://github.com/wbaek/torchskeleton)
class Mul(ch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x): return x * self.weight


class Flatten(ch.nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)


class Residual(ch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x): return x + self.module(x)


def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return ch.nn.Sequential(
        ch.nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                     stride=stride, padding=padding, groups=groups, bias=False),
        ch.nn.BatchNorm2d(channels_out),
        ch.nn.ReLU(inplace=True)
    )


def ResNet9(num_class, in_c=3):
    # num_class = 10
    model = ch.nn.Sequential(
        conv_bn(in_c, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(ch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        ch.nn.MaxPool2d(2),
        Residual(ch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        ch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        ch.nn.Linear(128, num_class, bias=False),
        Mul(0.2)
    )
    model = model.to(memory_format=ch.channels_last).cuda()
    return model


class CNN(nn.Module):
    def __init__(self, img_dim=(1,28,28), num_classes=10):
        super(CNN, self).__init__()
        assert img_dim[1] in [28,32]
        self.fe = torch.nn.Sequential(
            nn.Conv2d(img_dim[0], 32,3,1),  #26, 28
            nn.MaxPool2d(2,2),  # 13, 14
            nn.ReLU(),
            nn.Dropout(0.5), #0.5
            nn.Conv2d(32,64,3,1),   # 11, 12
            nn.MaxPool2d(2, 2), # 5, 6
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 128, 3, 1),   #3, 4
            nn.ReLU(),
            nn.Flatten()
        )
        if img_dim[1] == 28:
            self.cla = torch.nn.Sequential(
                nn.Linear(1152, 128),
                nn.ReLU(),
                #nn.Dropout2d(0.5),
                nn.Linear(128, num_classes),
            )
        elif img_dim[1] == 32:
            self.cla = torch.nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(),
                #nn.Dropout2d(0.5),
                nn.Linear(128, num_classes),

            )

    def forward(self, x):
        fes = self.fe(x)
        pred = self.cla(fes)
        return F.log_softmax(pred, dim=1)

    def pred(self, x):
        return F.softmax(self.cla(self.fe(x)), dim=1)


if __name__ == "__main__":
    model = ResNet9()
    x = torch.randn((32, 3, 32, 32)).cuda()
    y = model(x)
    print()
