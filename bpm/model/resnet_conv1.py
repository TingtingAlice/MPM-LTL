import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable

# resnet_conv1_pre_alpha_0.3


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    # original padding is 1; original dilation is 1
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # original padding is 1; original dilation is 1
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False,
                               dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, last_conv_stride=2, last_conv_dilation=1):

        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 63, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.alpha_maxpool_1 = nn.AvgPool2d(kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(63)

        # self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.alpha_maxpool_2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_conv_stride, dilation=last_conv_dilation)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        image_x = x[:, 0:3, :, :]
        a_x = x[:, 3:, :, :]

        #####################
        # use my maxpooling #
        #####################

        # too slow
        # x = self.conv1(image_x)
        # a_x = alpha_maxpool(a_x, kernel_size=7, stride=2, padding=3, limit=2)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        # a_x = alpha_maxpool(a_x, kernel_size=3, stride=2, padding=1, limit=2)
        # a_x.resize_(a_x.size()[0], 1, a_x.size()[1], a_x.size()[2])
        # a_x = a_x.type(torch.cuda.FloatTensor)
        # a_x = Variable(a_x)
        # x = torch.cat((x, a_x), 1)

        #####################
        # use nn.maxpooling #
        #####################

        x = self.conv1(image_x)

        a_x = torch.ge(a_x, 0.001).type(torch.cuda.FloatTensor)
        a_x = self.alpha_maxpool_1(a_x)
        a_x = torch.ge(a_x, 0.6).type(torch.cuda.FloatTensor)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        a_x = self.alpha_maxpool_2(a_x)
        a_x = torch.ge(a_x, 0.6).type(torch.cuda.FloatTensor)

        x = torch.cat((x, a_x), 1)

        ############
        # original #
        ############

        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def alpha_maxpool_gpu(input, kernel_size=3, stride=2, padding=1, limit=1):

    input = input.data

    print('input ', input.size())
    print('input ', type(input))

    dtype = torch.cuda.FloatTensor
    output = torch.cuda.FloatTensor(0)
    for i in range(input.size()[0]):
        # change size
        # img = np.zeros((input.size()[1] + padding * 2, input.size()[2] + padding * 2))
        img = torch.zeros(input.size()[2] + padding * 2, input.size()[3] + padding * 2).type(dtype)
        for j in range(input.size()[2]):
            for k in range(input.size()[3]):
                img[j + padding, k + padding] = input[i, 0, j, k]

        tmp = torch.cuda.FloatTensor(0)
        for j in range(0, img.size()[0], stride):
            tmp_row = torch.cuda.FloatTensor(0)
            if j + kernel_size > img.size()[0]:
                continue
            for k in range(0, img.size()[1], stride):
                if k + kernel_size > img.size()[1]:
                    continue
                # 255 count
                count = 0
                for j1 in range(j, j + kernel_size):
                    for k1 in range(k, k + kernel_size):
                        if img[j1, k1] != 0:
                            # see the value
                            count += 1
                tmp_col = torch.cuda.FloatTensor([0.])
                if count >= limit:
                    tmp_col = torch.cuda.FloatTensor([1.])
                if len(tmp_row.size()) == 0:
                    tmp_row = torch.cat((tmp_row, tmp_col), 0)  # 1 dim
                else:
                    tmp_row = torch.cat((tmp_row, tmp_col), 0)  # 1 dim
            if len(tmp.size()) == 0:
                # print('0, tmp')
                tmp = torch.cat((tmp, tmp_row), 0)  # 2 dim
                tmp = torch.unsqueeze(tmp, 0)
                # print('0, tmp', tmp.size())
            else:
                # print('tmp')
                tmp_row = torch.unsqueeze(tmp_row, 0)
                tmp = torch.cat((tmp, tmp_row), 0)  # 2 dim
                # print('tmp', tmp_row.size(), tmp.size())
        if len(output.size()) == 0:
            # print('0, output')
            output = torch.cat((output, tmp), 0)  # 3 dim
            output = torch.unsqueeze(output, 0)
            # print('0, output', output.size())
        else:
            # print('output')
            tmp = torch.unsqueeze(tmp, 0)
            output = torch.cat((output, tmp), 0)  # 3 dim
            # print('output', tmp.size(), output.size())
    # format
    print('output1 ', output.size())
    print('output1 ', type(output))
    print('end 1 ')
    output.resize_(output.size()[0], 1, output.size()[1], output.size()[2])
    output = Variable(output)
    print('end 2 ')

    print('output2 ', output.size())
    print('output2 ', type(output))

    return output



def remove_fc(state_dict):
    """Remove the fc layer parameters from state_dict."""
    for key, value in state_dict.items():
        if key.startswith('fc.'):
            del state_dict[key]
    return state_dict


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet18'])))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet34'])))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    model_dict = model.state_dict()

    if pretrained:
        pretrained_imagenet = remove_fc(model_zoo.load_url(model_urls['resnet50']))
        weight = model_dict.get('conv1.weight').numpy()
        weight[:63,:,:,:] = pretrained_imagenet['conv1.weight'][:63,:,:,:]
        pretrained_imagenet['conv1.weight'] = Parameter(torch.from_numpy(weight))

        weight = model_dict.get('bn1.running_mean').numpy()
        weight[:63,] = pretrained_imagenet['bn1.running_mean'][:63,]
        pretrained_imagenet['bn1.running_mean'] = Parameter(torch.from_numpy(weight))

        weight = model_dict.get('bn1.running_var').numpy()
        weight[:63,] = pretrained_imagenet['bn1.running_var'][:63,]
        pretrained_imagenet['bn1.running_var'] = Parameter(torch.from_numpy(weight))

        weight = model_dict.get('bn1.weight').numpy()
        weight[:63,] = pretrained_imagenet['bn1.weight'][:63,]
        pretrained_imagenet['bn1.weight'] = Parameter(torch.from_numpy(weight))

        weight = model_dict.get('bn1.bias').numpy()
        weight[:63,] = pretrained_imagenet['bn1.bias'][:63,]
        pretrained_imagenet['bn1.bias'] = Parameter(torch.from_numpy(weight))
        
        model_dict.update(pretrained_imagenet)
        print('model_dict  update')
        model.load_state_dict(model_dict)
        print('model.load_state_dict')
    else:

        model_new_dict = remove_fc(model_dict)
        model_dict.update(model_new_dict)
        model.load_state_dict(model_dict)

    print('end of resnet50 built')

    return model



def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(
            remove_fc(model_zoo.load_url(model_urls['resnet101'])))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(
            remove_fc(model_zoo.load_url(model_urls['resnet152'])))
    return model
