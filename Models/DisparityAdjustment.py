
import torch
import torch.nn as nn

import torchvision
class DisparityAdjustment(nn.Module):
    def __init__(self, device):
        super(DisparityAdjustment, self).__init__()
        self.netMaskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device).eval()

    def forward(self, orig_images, disparity_map):
        objPredictions = self.netMaskrcnn(orig_images[ :, [ 2, 0, 1 ], :, : ])
        tenAdjusted = torch.nn.functional.interpolate(input=disparity_map, size=(orig_images.shape[2], orig_images.shape[3]), mode='bilinear', align_corners=False)
        
        for i in range(len(objPredictions)):
            objPrediction = objPredictions[i]
            boolUsed = {}
            tenMasks = []
            for intMask in range(objPrediction['masks'].shape[0]):
                if intMask in boolUsed:
                    continue
                elif objPrediction['scores'][intMask].item() < 0.7:
                    continue
                elif objPrediction['labels'][intMask].item() not in [ 1, 3, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 ]:
                    continue
                boolUsed[intMask] = True
                tenMask = (objPrediction['masks'][(intMask + 0):(intMask + 1), :, :, :] > 0.5).float()

                if tenMask.sum().item() < 64:
                    continue

                for intMerge in range(objPrediction['masks'].shape[0]):
                    if intMerge in boolUsed:
                        continue

                    elif objPrediction['scores'][intMerge].item() < 0.7:
                        continue

                    elif objPrediction['labels'][intMerge].item() not in [ 2, 4, 27, 28, 31, 32, 33 ]:
                        continue

                    tenMerge = (objPrediction['masks'][(intMerge + 0):(intMerge + 1), :, :, :] > 0.5).float()

                    if ((tenMask + tenMerge) > 1.0).sum().item() < 0.03 * tenMerge.sum().item():
                        continue

                    boolUsed[intMerge] = True
                    tenMask = (tenMask + tenMerge).clip(0.0, 1.0)
                tenMasks.append(tenMask)

        

            for tenAdjust in tenMasks:
                tenPlane = tenAdjusted[i] * tenAdjust

                tenPlane = torch.nn.functional.max_pool2d(input=tenPlane.neg(), kernel_size=3, stride=1, padding=1).neg()
                tenPlane = torch.nn.functional.max_pool2d(input=tenPlane.neg(), kernel_size=3, stride=1, padding=1).neg()

                if tenPlane.sum().item() == 0: continue

                intLeft = (tenPlane.sum(2, True) > 0.0).flatten().nonzero()[0].item()
                intTop = (tenPlane.sum(3, True) > 0.0).flatten().nonzero()[0].item()
                intRight = (tenPlane.sum(2, True) > 0.0).flatten().nonzero()[-1].item()
                intBottom = (tenPlane.sum(3, True) > 0.0).flatten().nonzero()[-1].item()

                tenAdjusted[i] = ((1.0 - tenAdjust) * tenAdjusted[i]) + (tenAdjust * tenPlane[:, :, int(round(intTop + (0.97 * (intBottom - intTop)))):, :].max())

        return torch.nn.functional.interpolate(input=tenAdjusted, size=(disparity_map.shape[2], disparity_map.shape[3]), mode='bilinear', align_corners=False)

