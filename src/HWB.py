import torch
import torch.nn as nn
import torch.nn.functional as F

class HAB(nn.Module):
    def __init__(self, inplanes, planes):
        super(HAB, self).__init__()
        self.dilated_conv_1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1) 
        self.dilated_conv_2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=3, dilation=3)
        self.dilated_conv_3 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=9, dilation=9)
        self.dilated_conv_4 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=27, dilation=27)
        self.relu1 = nn.ReLU(True)
        self.relu2 = nn.ReLU(True)
        self.relu3 = nn.ReLU(True)
        self.relu4 = nn.ReLU(True)

        self.reshape = nn.Conv2d(160, 32, kernel_size=1,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()


    def forward(self, x):
        inp = x

        out1 = self.dilated_conv_1(x)
        out1 = self.relu1(out1)
        x1 = out1
        

        out2 = self.dilated_conv_2(x)
        out2 = self.relu2(out2)
        x2 = out2

        out3 = self.dilated_conv_3(x)
        out3 = self.relu3(out3)
        x3 = out3

        out4 = self.dilated_conv_4(x)
        out4 = self.relu4(out4)
        x4 = out4

        out = torch.cat([inp, x1, x2, x3, x4], 1)

        out = self.reshape(out)
        out = self.bn1(out)
        out = self.relu(out)

        return out


class SqEx_layer1(nn.Module):

    def __init__(self, n_features, reduction=16):
        super(SqEx_layer1, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        # self.hybrid_atn = HAB(32,32)
        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):
        # hyd = self.hybrid_atn(x)
        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y



class HAB2(nn.Module):
    def __init__(self, inplanes, planes):
        super(HAB2, self).__init__()
        self.dilated_conv_1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1) 
        self.dilated_conv_2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=3, dilation=3)
        self.dilated_conv_3 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=9, dilation=9)
        self.dilated_conv_4 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=27, dilation=27)
        self.relu1 = nn.ReLU(True)
        self.relu2 = nn.ReLU(True)
        self.relu3 = nn.ReLU(True)
        self.relu4 = nn.ReLU(True)

        self.reshape = nn.Conv2d(320, 64, kernel_size=1,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()


    def forward(self, x):
        inp = x

        out1 = self.dilated_conv_1(x)
        out1 = self.relu1(out1)
        x1 = out1
        

        out2 = self.dilated_conv_2(x)
        out2 = self.relu2(out2)
        x2 = out2

        out3 = self.dilated_conv_3(x)
        out3 = self.relu3(out3)
        x3 = out3

        out4 = self.dilated_conv_4(x)
        out4 = self.relu4(out4)
        x4 = out4

        out = torch.cat([inp, x1, x2, x3, x4], 1)

        out = self.reshape(out)
        out = self.bn1(out)
        out = self.relu(out)

        return out


class SqEx_layer2(nn.Module):

    def __init__(self, n_features, reduction=16):
        super(SqEx_layer2, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.hybrid_atn = HAB2(64,64)
        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):
        hyd = self.hybrid_atn(x)
        y = F.avg_pool2d(hyd, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y


class HAB3(nn.Module):
    def __init__(self, inplanes, planes):
        super(HAB3, self).__init__()
        self.dilated_conv_1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1) 
        self.dilated_conv_2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=3, dilation=3)
        self.dilated_conv_3 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=9, dilation=9)
        self.dilated_conv_4 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=27, dilation=27)
        self.relu1 = nn.ReLU(True)
        self.relu2 = nn.ReLU(True)
        self.relu3 = nn.ReLU(True)
        self.relu4 = nn.ReLU(True)

        self.reshape = nn.Conv2d(480, 96, kernel_size=1,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(96)
        self.relu = nn.ReLU()


    def forward(self, x):
        inp = x

        out1 = self.dilated_conv_1(x)
        out1 = self.relu1(out1)
        x1 = out1
        

        out2 = self.dilated_conv_2(x)
        out2 = self.relu2(out2)
        x2 = out2

        out3 = self.dilated_conv_3(x)
        out3 = self.relu3(out3)
        x3 = out3

        out4 = self.dilated_conv_4(x)
        out4 = self.relu4(out4)
        x4 = out4

        out = torch.cat([inp, x1, x2, x3, x4], 1)

        out = self.reshape(out)
        out = self.bn1(out)
        out = self.relu(out)

        return out


class SqEx_layer3(nn.Module):

    def __init__(self, n_features, reduction=16):
        super(SqEx_layer3, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.hybrid_atn = HAB3(96,96)
        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):
        hyd = self.hybrid_atn(x)
        y = F.avg_pool2d(hyd, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y



class HAB4(nn.Module):
    def __init__(self, inplanes, planes):
        super(HAB4, self).__init__()
        self.dilated_conv_1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1) 
        self.dilated_conv_2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=3, dilation=3)
        self.dilated_conv_3 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=9, dilation=9)
        self.dilated_conv_4 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=27, dilation=27)
        self.relu1 = nn.ReLU(True)
        self.relu2 = nn.ReLU(True)
        self.relu3 = nn.ReLU(True)
        self.relu4 = nn.ReLU(True)

        self.reshape = nn.Conv2d(640, 128, kernel_size=1,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()


    def forward(self, x):
        inp = x

        out1 = self.dilated_conv_1(x)
        out1 = self.relu1(out1)
        x1 = out1
        

        out2 = self.dilated_conv_2(x)
        out2 = self.relu2(out2)
        x2 = out2

        out3 = self.dilated_conv_3(x)
        out3 = self.relu3(out3)
        x3 = out3

        out4 = self.dilated_conv_4(x)
        out4 = self.relu4(out4)
        x4 = out4

        out = torch.cat([inp, x1, x2, x3, x4], 1)

        out = self.reshape(out)
        out = self.bn1(out)
        out = self.relu(out)

        return out


class SqEx_layer4(nn.Module):

    def __init__(self, n_features, reduction=16):
        super(SqEx_layer4, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.hybrid_atn = HAB4(128,128)
        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):
        hyd = self.hybrid_atn(x)
        y = F.avg_pool2d(hyd, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y

class HAB5(nn.Module):
    def __init__(self, inplanes, planes):
        super(HAB5, self).__init__()
        self.dilated_conv_1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1) 
        self.dilated_conv_2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=3, dilation=3)
        self.dilated_conv_3 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=9, dilation=9)
        self.dilated_conv_4 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=27, dilation=27)
        self.relu1 = nn.ReLU(True)
        self.relu2 = nn.ReLU(True)
        self.relu3 = nn.ReLU(True)
        self.relu4 = nn.ReLU(True)

        self.reshape = nn.Conv2d(3200, 640, kernel_size=1,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(640)
        self.relu = nn.ReLU()


    def forward(self, x):
        inp = x

        out1 = self.dilated_conv_1(x)
        out1 = self.relu1(out1)
        x1 = out1
        

        out2 = self.dilated_conv_2(x)
        out2 = self.relu2(out2)
        x2 = out2

        out3 = self.dilated_conv_3(x)
        out3 = self.relu3(out3)
        x3 = out3

        out4 = self.dilated_conv_4(x)
        out4 = self.relu4(out4)
        x4 = out4

        out = torch.cat([inp, x1, x2, x3, x4], 1)

        out = self.reshape(out)
        out = self.bn1(out)
        out = self.relu(out)

        return out


class SqEx_layer5(nn.Module):

    def __init__(self, n_features, reduction=16):
        super(SqEx_layer5, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.hybrid_atn = HAB5(640,640)
        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):
        hyd = self.hybrid_atn(x)
        y = F.avg_pool2d(hyd, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y