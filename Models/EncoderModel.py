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
    
class EncoderModelConvNeXt(nn.Module):
    def __init__(self):
        super(EncoderModelConvNeXt, self).__init__()
        self.numChannels = np.array([96, 96, 192, 384, 768])
        self.encoder = models.convnext_tiny(pretrained=True)
        self.layers = {layer[0]:layer[1] for layer in self.encoder.named_modules() if layer[0].startswith('features') and len(layer[0].split("."))==2}

    def forward(self, inputImg):
        self.features = []
        out = (inputImg - 0.45)/0.225
        out = self.layers['features.0'](out)
        self.features.append(nn.functional.interpolate(out, scale_factor=2, mode="nearest"))
        out = self.layers['features.1'](out)
        self.features.append(out)
        for layerIdx in range(2, 7, 2):
            out = self.layers['features.{}'.format(layerIdx+1)](self.layers['features.{}'.format(layerIdx)](out))
            self.features.append(out)
        return self.features