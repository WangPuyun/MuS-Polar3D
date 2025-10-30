import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class Network(nn.Module):
    def __init__(self, skip_res=True):  # normal_map 3通道
        super(Network, self).__init__()
        self.skip_res = skip_res

        self.encoder_pre = Extractor(17, 4)
        self.encoder_1 = ConvDown(4, 32)
        self.encoder_2 = ConvDown(32, 64)
        self.encoder_3 = ConvDown(64, 128)
        self.encoder_4 = ConvDown(128, 256)
        self.encoder_5 = ConvDown(256, 512)

        self.deconder_1 = UpSam(512, 512)
        self.deconder_2 = UpSam(1024, 256)
        self.deconder_3 = UpSam(512, 128)
        self.deconder_4 = UpSam(256, 64)
        self.deconder_5 = UpSam(128, 32)
        self.deconder_end = FinalLayer(64, 3)

    # 整个流程
    def forward(self, input, image):

        input = self.encoder_pre(input)

        x1 = self.encoder_1(input)
        x2 = self.encoder_2(x1)
        x3 = self.encoder_3(x2)
        x4 = self.encoder_4(x3)
        x5 = self.encoder_5(x4)

        x = self.deconder_1(x5, x5, image)
        x = self.deconder_2(x, x4, image)
        x = self.deconder_3(x, x3, image)
        x = self.deconder_4(x, x2, image)
        x = self.deconder_5(x, x1, image)
        x = self.deconder_end(x)

        return x

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)

class Extractor(nn.Module):
    ########  Polarized Feature Extractor  ########
    def __init__(self, inplanes, planes):
        super(Extractor, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0)
        self.norm = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class ConvDown(nn.Module):
    ########  Convolutional Downsample  ########
    def __init__(self, inplanes, planes, padding=1):
        super(ConvDown, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=2, padding=padding)  # 3*3
        self.In1 = nn.InstanceNorm2d(planes)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=padding)  # 3*3
        self.In2 = nn.InstanceNorm2d(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.In1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.In2(out)
        out = self.relu(out)
        return out


class UpSam(nn.Module):
    ########  Upsample  ########
    def __init__(self, inplanes, planes, padding=1):  # 初始化父类的属性 Numberh维度
        super(UpSam, self).__init__()  # 特殊函数super，能够调用父类的方法
        self.BupSam = nn.UpsamplingBilinear2d(scale_factor=2)  # 参数只要一个 （scale_factor或者size)
        self.snorm1 = SPADE(inplanes)
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=padding)  # 3*3
        self.snorm2 = SPADE(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=padding)  # 3*3

    def forward(self, x, x_pre, resize_image):
        out = self.BupSam(x)
        out = self.snorm1(out, resize_image)
        out = self.relu(out)
        out1 = self.conv1(out)
        out = self.snorm2(out1, resize_image)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + out1
        x_pre = F.interpolate(x_pre, size=(out.size(2), out.size(3)), mode='nearest')  # mode
        out = torch.cat([out, x_pre], dim=1)
        return out

class SPADE(nn.Module):
    def __init__(self, out_channels):
        super(SPADE, self).__init__()
        self.norm = nn.BatchNorm2d(out_channels, affine=False)
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(4, out_channels, 3, 1, 1)),
            nn.ReLU()
        )
        self.conv_gamma = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
        self.conv_beta = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1))

    def forward(self, x, polar_image):
        polar_image = F.interpolate(polar_image, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        polar_image = self.conv(polar_image)
        return self.norm(x) * self.conv_gamma(polar_image) + self.conv_beta(polar_image) + self.norm(x)

class FinalLayer(nn.Module):
    ########  Final Layer  ########
    def __init__(self, inplanes, planes):
        super(FinalLayer, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1)
        self.In = nn.InstanceNorm2d(planes)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.In(out)
        out = self.relu(out)
        out += x
        out = self.conv2(out)
        return out

net = Network()
net.apply(initialize_weights)

if __name__ == '__main__':
    Net = Network()
    input1 = torch.ones([8, 13, 256, 256])
    input2 = torch.ones([8, 4, 256, 256])
    output = Net(input1, input2)
    print(output.shape)

