import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import config
from utils import RGB_to_L, L_to_RGB
from modules import Cell

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

class Classifier(nn.Module):
    def __init__(self, classes_size):
        super(Classifier, self).__init__()
        self.classes_size = classes_size
        self.classifier_info = self.make_classifier_info()
        self.classifier = self.make_classifier()
        
    def make_classifier_info(self):
        classifier_info = [ 
        {'input_size':512,'output_size':self.classes_size,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':'none','raw':False},
        {'cell':'PoolCell','mode':'avg','output_size':1}
        ]
        return classifier_info

    def make_classifier(self):
        classifier = nn.ModuleList([])
        for i in range(len(self.classifier_info)):
            classifier.append(Cell(self.classifier_info[i]))
        return classifier
        
    def classification_loss_fn(self, input, output, protocol):
        if(protocol['loss_mode']['classification'] == 'ce'):
            loss_fn = F.cross_entropy
        else:
            raise ValueError('classification loss mode not supported')
        if(protocol['tuning_param']['classification'] > 0):
            loss = loss_fn(output['classification'],input['label'],reduction='mean')
            loss = loss.mean()
        else:
            loss = torch.tensor(0,device=device,dtype=torch.float32) 
        return loss
        
    def forward(self, input, protocol):
        x = input
        for i in range(len(self.classifier)):
            x = self.classifier[i](x)        
        x = x.view(x.size(0),self.classes_size)
        return x
        
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, classes_size=10, if_classify = True):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layers = self.make_layers(block, num_blocks, stride)
        self.classifier = Classifier(classes_size)
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
        
    def forward(self, input, protocol):
        if(self.if_classify):
            x = L_to_RGB(input['img']) if (protocol['img_mode'] == 'L') else input['img']
            x = F.relu(self.bn1(self.conv1(x)))
            for i in range(len(self.layers)):
                x = self.layers[i](x)
            output['compression']['code'] = x
            output['classification'] = self.classifier(x,protocol)
            output['loss'] = protocol['tuning_param']['classification']*self.classifier.classification_loss_fn(input,output,protocol)
            return output
        else:
            output = {'loss':torch.tensor(0,device=device,dtype=torch.float32),
            'compression':{'img':torch.tensor(0,device=device,dtype=torch.float32),'code':[]},
            'classification':torch.tensor(0,device=device,dtype=torch.float32)}
            x = L_to_RGB(input) if (protocol['img_mode'] == 'L') else input
            x = F.relu(self.bn1(self.conv1(x)))
            for i in range(len(self.layers)):
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
