from Trainers.BaseTrainer import BaseTrainer
from Models.DisparityAdjustmentV2 import DisparityAdjustment
from Models.IntrinsicsModel import IntrinsicsModel

import torch.nn.functional as F

class IntrinsicsTrainer(BaseTrainer):
    def __init__(self,config):
        super().__init__(config)

        self.models['intrinsics'] = IntrinsicsModel().to(self.device)
        # self.models["intrinsics"] = self.models["intrinsics"].to(self.device)

    def predictPoses(self, inputs, features):
        outputs = {}
        poseFeatures = {fi: features[fi] for fi in self.frameIdxs}
        for fi in self.frameIdxs[1:]:
            if fi < 0:
                poseInputs = [poseFeatures[fi], poseFeatures[0]]
            else:
                poseInputs = [poseFeatures[0], poseFeatures[fi]]
            axisangle, translation,bottleneck = self.models["pose"](poseInputs)
            outputs[("axisangle", 0, fi)] = axisangle
            outputs[("translation", 0, fi)] = translation
            outputs[("cam_T_cam", 0, fi)] = self.transformParameters(axisangle[:, 0], translation[:, 0], invert=(fi<0))
            outputs[("bottleneck", 0, fi)] = bottleneck
        return outputs

    def generateImagePredictions(self, inputs, outputs):
        for scale in range(self.numScales):

            # # Disparity Adjustment
            # orig_scaled_images = inputs[("color", 0, scale)]
            # outputs[("disp", scale)] = self.disparityadjustment(orig_scaled_images,outputs[("disp", scale)])

            disp = outputs[("disp", scale)]
            
            disp = F.interpolate(disp, [self.height, self.width], mode="bilinear",
                                 align_corners=False)

            sourceScale = 0
            _, depth = self.dispToDepth(disp, 0.1, 100.0)
            outputs[("depth", 0, scale)] = depth
            for i, frameIdx in enumerate(self.frameIdxs[1:]):
                T = outputs[("cam_T_cam", 0, frameIdx)]
                bottleneck = outputs[("bottleneck", 0, frameIdx)]

                # predicit camera intrinsics
                outputs[("K", 0, frameIdx)] = self.models['intrinsics'](bottleneck, self.width, self.height)
                outputs[("inv_K", 0, frameIdx)] = torch.inverse(outputs[("K", 0, frameIdx)])

                cameraPoints = self.backprojectDepth[sourceScale](depth, outputs[("inv_K", 0, frameIdx)])
                pixelCoordinates = self.project3d[sourceScale](cameraPoints, outputs[("K", 0, frameIdx)], T)
                
                outputs[("sample", frameIdx, scale)] = pixelCoordinates
                outputs[("color", frameIdx, scale)] = F.grid_sample(inputs[("color", frameIdx, sourceScale)],
                                                                    outputs[(("sample", frameIdx, scale))],
                                                                    padding_mode="border")
                outputs[("color_identity", frameIdx, scale)] = inputs[("color", frameIdx, sourceScale)]