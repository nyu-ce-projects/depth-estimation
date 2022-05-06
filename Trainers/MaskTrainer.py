
from Trainers.BaseTrainer import BaseTrainer
from Models.DisparityAdjustmentV2 import DisparityAdjustment

import torch.nn.functional as F


class MaskTrainer(BaseTrainer):
    def __init__(self,config):
        super().__init__(config)

        #Disparity Adjustment Model
        self.disparityadjustment = DisparityAdjustment(self.device)

    def generateImagePredictions(self, inputs, outputs):
        for scale in range(self.numScales):

            # Disparity Adjustment
            orig_scaled_images = inputs[("color", 0, scale)]
            outputs[("disp", scale)] = self.disparityadjustment(orig_scaled_images,outputs[("disp", scale)])

            disp = outputs[("disp", scale)]
            
            disp = F.interpolate(disp, [self.height, self.width], mode="bilinear",
                                 align_corners=False)

            sourceScale = 0
            _, depth = self.dispToDepth(disp, 0.1, 100.0)
            outputs[("depth", 0, scale)] = depth
            for i, frameIdx in enumerate(self.frameIdxs[1:]):
                T = outputs[("cam_T_cam", frameIdx, 0)]
                cameraPoints = self.backprojectDepth[sourceScale](depth, inputs[("inv_K", sourceScale)])
                pixelCoordinates = self.project3d[sourceScale](cameraPoints, inputs[("K", sourceScale)], T)
                outputs[("sample", frameIdx, scale)] = pixelCoordinates
                outputs[("color", frameIdx, scale)] = F.grid_sample(inputs[("color", frameIdx, sourceScale)],
                                                                    outputs[(("sample", frameIdx, scale))],
                                                                    padding_mode="border")
                outputs[("color_identity", frameIdx, scale)] = inputs[("color", frameIdx, sourceScale)]