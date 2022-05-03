from Trainers.BaseTrainer import BaseTrainer
from Models.CameraNet import CameraNet

import torch
import torch.nn.functional as F


class CamNetTrainer(BaseTrainer):
    def __init__(self,config):
        super().__init__(config)

        # Camera Net Model - Pose Estimation + Camera Intrinsics
        self.models["pose"] = CameraNet(self.models["encoder"].numChannels[0], h=self.height//2,
                                        w=self.width//2, refine=False)
        self.models["pose"] = self.models["pose"].to(self.device)

        # Call to add new trainable parameters from CameraNet
        self.setupOptimizer()


    def predictPoses(self, inputs, features):
        outputs = {}
        poseFeatures = {fi: features[fi] for fi in self.frameIdxs}
        for fi in self.frameIdxs[1:]:
            if fi < 0:
                poseInputs = [poseFeatures[fi][0], poseFeatures[0][0]]
            else:
                poseInputs = [poseFeatures[0][0], poseFeatures[fi][0]]
            poseInputs = torch.cat(poseInput, dim=1)
            axisangle, translation, _, K = self.models["pose"](poseInputs)
            outputs[("axisangle", 0, fi)] = axisangle
            outputs[("translation", 0, fi)] = translation
            outputs[("cam_T_cam", 0, fi)] = self.transformParameters(axisangle[:, 0], translation[:, 0], invert=(fi<0))
            for scaleNum in range(self.numScales):
                outputs[("K", scaleNum, fi)] = K//(2*scaleNum)
                outputs[("inv_K", scaleNum, fi)] = torch.linalg.pinv(outputs[("K", scaleNum, fi)])
        return outputs


    def generateImagePredictions(self, inputs, outputs):
        for scale in range(self.numScales):
            disp = outputs[("disp", scale)]

            disp = F.interpolate(disp, [self.height, self.width], mode="bilinear",
                                 align_corners=False)

            sourceScale = 0
            _, depth = self.dispToDepth(disp, 0.1, 100.0)
            outputs[("depth", 0, scale)] = depth
            for i, frameIdx in enumerate(self.frameIdxs[1:]):
                T = outputs[("cam_T_cam", 0, frameIdx)]
                cameraPoints = self.backprojectDepth[sourceScale](depth, outputs[("inv_K", sourceScale, frameIdx)])
                pixelCoordinates = self.project3d[sourceScale](cameraPoints, outputs[("K", sourceScale, frameIdx)], T)
                outputs[("sample", frameIdx, scale)] = pixelCoordinates
                outputs[("color", frameIdx, scale)] = F.grid_sample(inputs[("color", frameIdx, sourceScale)],
                                                                    outputs[(("sample", frameIdx, scale))],
                                                                    padding_mode="border")
                outputs[("color_identity", frameIdx, scale)] = inputs[("color", frameIdx, sourceScale)]
