from venv import create
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import Tensor


class LeNet(nn.Module):
    def __init__(self, classes) -> None:
        super(LeNet, self).__init__()

        activation_class = nn.Sigmoid

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            activation_class(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, padding=0),
            activation_class(),
            nn.MaxPool2d(2, stride=2),
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            activation_class(),
            nn.Linear(120, 84),
            activation_class(),
            nn.Linear(84, classes)
        )
    
    def forward(self, x: Tensor):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return F.softmax(x, dim=1)

class AlexNet(nn.Module):
    def __init__(self, classes=100) -> None:
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(4096, classes)
        )
    
    def forward(self, x: Tensor):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

class AlexNet32(nn.Module):
    def __init__(self, classes=100) -> None:
        super(AlexNet32, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(256*1*1, 4096),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(4096, classes)
        )
    
    def forward(self, x: Tensor):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

class VGG(nn.Module):
    def __init__(self, classes):
        super(VGG, self).__init__()
        conv_args = dict(
            kernel_size=3,
            stride=1,
            padding=1,
        )

        pool_args = dict(
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, **conv_args),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(**pool_args),
            nn.Conv2d(64, 128, **conv_args),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(**pool_args),
            nn.Conv2d(128, 256, **conv_args),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, **conv_args),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(**pool_args),
            nn.Conv2d(256, 512, **conv_args),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, **conv_args),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(**pool_args),
            nn.Conv2d(512, 512, **conv_args),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, **conv_args),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(**pool_args),
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, classes)
        )
    
    def forward(self, x: Tensor):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return F.softmax(x, dim=1)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.contr1 = Contract(1, 64)
        self.contr2 = Contract(64, 128)
        self.contr3 = Contract(128, 256)
        self.contr4 = Contract(256, 512)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottom = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3),
        )

        self.expand1 = Expand(1024, 512, 56)
        self.expand2 = Expand(512, 256, 104)
        self.expand3 = Expand(256, 128, 200)
        self.expand4 = Expand(128, 64, 392)
        self.final_conv = nn.Conv2d(64, 2, kernel_size=1)

    
    def forward(self, x: Tensor):
        x1 = self.contr1(x)
        x2 = self.contr2(self.maxpool(x1))
        x3 = self.contr3(self.maxpool(x2))
        x4 = self.contr4(self.maxpool(x3))
        x = self.bottom(x4)
        x = self.expand1(x4, x)
        x = self.expand2(x3, x)
        x = self.expand3(x2, x)
        x = self.expand4(x1, x)
        x = self.final_conv(x)
        return x


class Contract(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Contract, self).__init__()

        self.contract = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
    )
    
    def forward(self, x: Tensor):
        return self.contract(x)

class Expand(nn.Module):
    def __init__(self, in_channels, out_channels, img_size):
        super(Expand, self).__init__()

        self.crop = transforms.CenterCrop(img_size)

        self.up_conv = nn.Conv2d(in_channels, out_channels, kernel_size=2)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor, y: Tensor):
        x = self.crop(x)
        y = self.up_conv(y)
        z = torch.cat((x, y), dim=1)
        z = self.conv(z)
        return z

if __name__ == '__main__':
    from torchinfo import summary

    model = UNet()
    summary(model, input_size=(32, 1, 572, 572))
