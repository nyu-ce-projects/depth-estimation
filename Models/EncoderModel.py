import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

class EncoderModel(nn.Module):
    def __init__(self, numLayers):
        super(EncoderModel, self).__init__()
        self.numChannels = np.array([64, 64, 128, 256, 512])
        if numLayers > 34:
            self.numChannels[1:] *= 4
        modelsDict =    {
                        18  :   models.resnet18,
                        34  :   models.resnet34,
                        50  :   models.resnet50,
                        101 :   models.resnet101,
                        152 :   models.resnet152
                        }
        self.encoder = modelsDict[numLayers](True)

    def forward(self, inputImg):
        self.features = []
        out = (inputImg - 0.45)/0.225
        out = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(out)))
        self.features.append(out)
        out = self.encoder.layer1(self.encoder.maxpool(out))
        self.features.append(out)
        for layer in ["2", "3", "4"]:
            out = eval("self.encoder.layer"+layer+"(out)")
            self.features.append(out)
        return self.features

class MultiImageEncoderModel(models.ResNet):
    def __init__(self, numLayers):
        super(MultiImageEncoderModel, self).__init__(models.resnet.Bottleneck, [3, 4, 6, 3], 1000)
        self.numChannels = np.array([64, 64, 128, 256, 512])
        if numLayers > 34:
            self.numChannels[1:] *= 4
        self.inplanes = 64
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(models.resnet.Bottleneck, 64, 3)
        self.layer2 = self._make_layer(models.resnet.Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(models.resnet.Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(models.resnet.Bottleneck, 512, 3, stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(numLayers)])
        loaded['conv1.weight'] = torch.cat([loaded['conv1.weight']] * 2, 1)/2
        self.load_state_dict(loaded)

    def forward(self, inputImg):
        self.features = []
        out = (inputImg - 0.45)/0.225
        out = self.relu(self.bn1(self.conv1(out)))
        self.features.append(out)
        out = self.layer1(self.maxpool(out))
        self.features.append(out)
        for layer in ["2", "3", "4"]:
            out = eval("self.layer"+layer+"(out)")
            self.features.append(out)
        return self.features

