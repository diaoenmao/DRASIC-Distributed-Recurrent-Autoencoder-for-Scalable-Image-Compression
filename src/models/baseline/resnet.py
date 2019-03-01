import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import config
from utils import RGB_to_L, L_to_RGB
config.init()
device = config.PARAM['device']
max_depth = 3
max_channel = 512
n_channels = [128, 256, max_channel, max_channel, max_channel, max_channel]
stride = [1, 2, 2, 2, 2, 2]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ClassifierCell(nn.Module):
    def __init__(self, n_class):
        super(ClassifierCell, self).__init__()
        self.n_class = n_class
        self.fc = nn.Linear(max_channel, n_class)
        
    def forward(self, input):
        x = F.adaptive_avg_pool2d(input, 1).view(input.size(0),-1)
        x = self.fc(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, classes_size=10, if_classify = True):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layers = self.make_layers(block, num_blocks, stride)
        self.classifier = ClassifierCell(classes_size)
        self.if_classify = if_classify
        
    def make_layers(self,block,num_blocks,stride):
        module_list = []
        in_channels = 64
        for i in range(max_depth):
            strides = [stride[i]] + [1]*(num_blocks[i]-1)
            layers = []
            for s in strides:
                layers.append(block(in_channels, n_channels[i], s))
                in_channels = n_channels[i] * block.expansion
            module_list.append(nn.Sequential(*layers))
        layers = nn.ModuleList(module_list)
        return layers

    def classification_loss_fn(self, output, target):
        loss = F.cross_entropy(output,target)
        return loss
        
    def forward(self, input, protocol):
        mode = protocol['mode']
        depth = protocol['depth']
        if(self.if_classify):
            output = {}
            img = input['img']
            label = input['label']
            x = L_to_RGB(img) if (mode == 'L') else img
            x = F.relu(self.bn1(self.conv1(x)))
            for i in range(depth):
                x = self.layers[i](x)
            x = self.classifier(x)
            loss = self.classification_loss_fn(x, label)
            output['classification'] = x
            output['loss'] = loss
            return output
        else:
            x = L_to_RGB(input) if (mode == 'L') else input
            x = F.relu(self.bn1(self.conv1(x)))
            for i in range(depth):
                x = self.layers[i](x)
            return x


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3, 3, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3, 3, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3, 3, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3, 3, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
