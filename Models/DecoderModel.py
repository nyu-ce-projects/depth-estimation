import torch
import torch.nn as nn
import numpy as np

from collections import OrderedDict

class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, activation=True):
        super(ConvBlock, self).__init__()
        self.activation = activation
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=3)
        self.act = nn.ELU()

    def forward(self, x):
        if self.activation:
            out = self.act(self.conv(self.pad(x)))
        else:
            out = self.conv(self.pad(x))
        return out

class DecoderModel(nn.Module):
    def __init__(self, numChannelsEncoder):
        super(DecoderModel, self).__init__()
        self.numChannelsEncoder = numChannelsEncoder
        self.numChannelsDecoder = np.array([16, 32, 64, 128, 256])
        self.convs = OrderedDict()
        for layer in range(4, -1, -1):
            inChannels = self.numChannelsEncoder[-1] if layer == 4 else self.numChannelsDecoder[layer+1]
            outChannels = self.numChannelsDecoder[layer]
            self.convs[("upconv", layer, 0)] = ConvBlock(inChannels, outChannels, activation=True)
            inChannels = self.numChannelsDecoder[layer]
            if layer > 0:
                inChannels += self.numChannelsEncoder[layer-1]
            outChannels = self.numChannelsDecoder[layer]
            self.convs[("upconv", layer, 1)] = ConvBlock(inChannels, outChannels, activation=True)
        for scale in range(4):
            self.convs[("dispconv", scale)] = ConvBlock(self.numChannelsDecoder[scale], 1, activation=False)
        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, inputFeatures):
        self.outputs = {}
        x = inputFeatures[-1]
        for layer in range(4, -1, -1):
            x = self.convs[("upconv", layer, 0)](x)
            x = [nn.functional.interpolate(x, scale_factor=2, mode="nearest")]
            if layer > 0:
                x += [inputFeatures[layer-1]]
            x = torch.cat(x, dim=1)
            x = self.convs[("upconv", layer, 1)](x)
            if layer < 4:
                out = self.convs[("dispconv", layer)](x)
                self.outputs[("disp", layer)] = torch.sigmoid(out)
        return self.outputs
