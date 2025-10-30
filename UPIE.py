import torch
import torch.nn as nn
import torch.nn.functional as F


class MLFE(nn.Module):
    def __init__(self):
        super(MLFE, self).__init__()
        # LFEB
        self.conv1 = BasicConv(4, 16, 1, 1)
        self.conv2 = BasicConv(16, 16, 3, 1)
        self.conv3 = BasicConv(16, 16, 3, 1, dila=1)
        self.conv4 = BasicConv(16, 16, 3, 1, dila=3)
        self.conv5 = BasicConv(16, 16, 3, 1, dila=5)
        self.conv6 = BasicConv(48, 16, 1, 1)
        # EB
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # Pooling for downsampling
        self.Conv1 = BasicConv(16, 32, 3, 1)
        self.Conv2 = BasicConv(32, 64, 3, 1)
        self.Conv3 = BasicConv(64, 128, 3, 1)
        self.Conv4 = BasicConv(128, 256, 3, 1)
        self.Conv5 = BasicConv(256, 512, 3, 1)

        # Apply Dropout2d at the encoder-decoder junction (bottleneck)
        self.dropout = nn.Dropout2d(p=0.5)

        # DB
        self.CONV1 = BasicConv(512, 256, 3, 1)
        self.Tconv1 = BasicConv(512, 256, kernel_size=2, activation=True, stride=2, transpose=True)  # 反卷积用于上采样
        self.CONV2 = BasicConv(256, 128, 3, 1)
        self.Tconv2 = BasicConv(256, 128, kernel_size=2, activation=True, stride=2, transpose=True)
        self.CONV3 = BasicConv(128, 64, 3, 1)
        self.Tconv3 = BasicConv(128, 64, kernel_size=2, activation=True, stride=2, transpose=True)
        self.CONV4 = BasicConv(64, 32, 3, 1)
        self.Tconv4 = BasicConv(64, 32, kernel_size=2, activation=True, stride=2, transpose=True)
        self.CONV5 = BasicConv(32, 16, 3, 1)
        # Final output layer
        self.final_conv = BasicConv(16, 4, 3, 1, norm=False, activation=True, is_last=True)

    def forward(self, x):
        # -----------Local Feature Extraction Block(LFEB)-----------
        x1 = self.conv1(x)
        x2_1, x2_2 = self.conv3(x1), self.conv2(x1)
        x3_1, x3_2 = self.conv4(x2_2), self.conv2(x2_2)
        x4_1 = self.conv5(x3_2)
        x5 = torch.cat((x2_1, x3_1, x4_1), dim=1)
        x6 = self.conv6(x5)
        x7 = x6 + x1  # Residual addition

        # ------------------ Encoder Block (EB) ------------------
        e1 = self.Conv1(x7)
        e2 = self.Conv2(self.Maxpool(e1))
        e3 = self.Conv3(self.Maxpool(e2))
        e4 = self.Conv4(self.Maxpool(e3))
        e5 = self.Conv5(self.Maxpool(e4))

        # ---- Apply dropout in the bottleneck ----
        e5 = self.dropout(e5)

        # ------------------ Decoder Block (DB) ------------------
        # d5
        d5_1 = self.Tconv1(e5)
        d5 = self.CONV1(torch.cat((d5_1, e4), dim=1))
        # d4
        d4_1 = self.Tconv2(d5)
        d4 = self.CONV2(torch.cat((d4_1, e3), dim=1))
        # d3
        d3_1 = self.Tconv3(d4)
        d3 = self.CONV3(torch.cat((d3_1, e2), dim=1))
        # d2
        d2_1 = self.Tconv4(d3)
        d2 = self.CONV4(torch.cat((d2_1, e1), dim=1))
        # d1
        d1 = self.CONV5(d2)

        # ------------------ Final Output Layer ------------------
        output = self.final_conv(d1)

        return output


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, dila=1, bias=True, norm=True, activation=True,
                 transpose=False, is_last=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = ((kernel_size - 1) * dila) // 2

        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        elif is_last:
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                                    dilation=dila))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                                    dilation=dila))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if activation:
            layers.append(nn.ReLU(inplace=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
