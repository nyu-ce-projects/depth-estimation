from Trainers.BaseTrainer import BaseTrainer
from Models.DisparityAdjustmentV2 import DisparityAdjustment
from Models.IntrinsicsModel import IntrinsicsModel
from Models.DecoderModel import DepthDecoderModelESPCN

import torch
import torch.nn.functional as F

class MaskCamlessTrainer(BaseTrainer):
    def __init__(self,config):
        super().__init__(config)

        if config['model_name']=='MASKCAMLESS_ESPCN':
            self.models["decoder"] = DepthDecoderModelESPCN(self.models["encoder"].numChannels)
            self.models["decoder"] = self.models["decoder"].to(self.device)

        #Disparity Adjustment Model
        self.disparityadjustment = DisparityAdjustment(self.device)

        self.models['intrinsics'] = IntrinsicsModel()  #.to(self.device)
        self.models["intrinsics"] = self.models["intrinsics"].to(self.device)

        self.setupOptimizer()

    def predictPoses(self, inputs, features):
        outputs = {}
        poseFeatures = {fi: features[fi] for fi in self.frameIdxs}
        for fi in self.frameIdxs[1:]:
            if fi < 0:
                poseInputs = [poseFeatures[fi], poseFeatures[0]]
            else:
                poseInputs = [poseFeatures[0], poseFeatures[fi]]
            axisangle, translation,bottleneck = self.models["pose"](poseInputs)
            outputs[("axisangle", fi, 0)] = axisangle
            outputs[("translation", fi, 0)] = translation
            outputs[("cam_T_cam", fi, 0)] = self.transformParameters(axisangle[:, 0], translation[:, 0], invert=(fi<0))
            outputs[("bottleneck", fi, 0)] = bottleneck
        return outputs

    def generateImagePredictions(self, inputs, outputs):
        for scale in range(self.numScales):
            # Disparity Adjustment
            orig_scaled_images = inputs[("color", 0, scale)]
            disp = self.disparityadjustment(orig_scaled_images,outputs[("disp", scale)])

            # disp = outputs[("disp", scale)]
            disp = F.interpolate(disp, [self.height, self.width], mode="bilinear",align_corners=False)
            sourceScale = 0
            _, depth = self.dispToDepth(disp, 0.1, 100.0)
            outputs[("depth", 0, scale)] = depth
            for i, frameIdx in enumerate(self.frameIdxs[1:]):
                bottleneck = outputs[("bottleneck", frameIdx, 0)]
                T = outputs[("cam_T_cam", frameIdx, 0)]

                # predicit camera intrinsics
                outputs[("K", frameIdx, 0)] = self.models['intrinsics'](bottleneck, self.width, self.height)
                outputs[("inv_K", frameIdx, 0)] = torch.inverse(outputs[("K", frameIdx, 0)])

                cameraPoints = self.backprojectDepth[sourceScale](depth, outputs[("inv_K", frameIdx, 0)])
                pixelCoordinates = self.project3d[sourceScale](cameraPoints, outputs[("K", frameIdx, 0)], T)
                outputs[("sample", frameIdx, scale)] = pixelCoordinates
                outputs[("color", frameIdx, scale)] = F.grid_sample(inputs[("color", frameIdx, sourceScale)],outputs[(("sample", frameIdx, scale))],padding_mode="border", align_corners=False)
                outputs[("color_identity", frameIdx, scale)] = inputs[("color", frameIdx, sourceScale)]   