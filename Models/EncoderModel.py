import numpy as np
import torch.nn as nn
import torchvision.models as models

class EncoderModel(nn.Module):
    def __init__(self, numLayers):
        super(EncoderModel, self).__init__()
        self.numChannels = np.array([64, 64, 128, 256, 512])
        modelsDict =    {
                        18  :   models.resnet18,
                        34  :   models.resnet34,
                        50  :   models.resnet50,
                        101 :   models.resnet101,
                        152 :   models.resnet152
                        }
        if numLayers > 34:
            self.numChannels[1:] *= 4
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
