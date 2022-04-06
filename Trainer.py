import os
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from Models.EncoderModel import EncoderModel, MultiImageEncoderModel
from Models.DecoderModel import DepthDecoderModel, PoseDecoderModel
from Models.BackprojectDepth import BackprojectDepth
from Models.Project3D import Project3D
from Dataset.KITTI import KITTI

class Trainer:
    def __init__(self, LR=0.001, batchSize=48, epochs=20, height=192, width=640, frameIdxs=[0, -1, 1],
                 scales=[0, 1, 2, 3]):
        self.LR = LR
        self.batchSize = batchSize
        self.epochs = epochs
        self.height = height
        self.width = width
        self.frameIdxs = frameIdxs
        self.numScales = len(scales)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.trainableParameters = []
        self.models["encoder"] = EncoderModel(50)
        self.models["encoder"] = self.models["encoder"].to(self.device)
        self.trainableParameters += list(self.models["encoder"].parameters())
        self.models["decoder"] = DepthDecoderModel(self.models["encoder"].numChannels)
        self.models["decoder"] = self.models["decoder"].to(self.device)
        self.trainableParameters += list(self.models["decoder"].parameters())
        self.models["pose_encoder"] = MultiImageEncoderModel(50)
        self.models["pose_encoder"] = self.models["pose_encoder"].to(self.device)
        self.trainableParameters += list(self.models["pose_encoder"].parameters())
        self.models["pose"] = PoseDecoderModel(self.models["pose_encoder"].numChannels)
        self.models["pose"] = self.models["pose"].to(self.device)
        self.trainableParameters += list(self.models["pose"].parameters())
        self.optimizer = optim.Adam(self.trainableParameters, lr=self.LR)
        self.lrScheduler = optim.lr_scheduler.StepLR(self.optimizer, 15, 0.1)
        self.loadDataset()
        self.depthMetricNames = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        self.backprojectDepth = {}
        self.project3d = {}
        for scale in range(self.numScales):
            h = self.height // (2**scale)
            w = self.width // (2**scale)
            self.backprojectDepth[scale] = BackprojectDepth(self.batchSize, h, w)
            self.backprojectDepth[scale] = self.backprojectDepth[scale].to(self.device)
            self.project3d[scale] = Project3D(self.batchSize, h, w)
            self.project3d[scale] = self.project3d[scale].to(self.device)

    def readlines(self, path):
        with open(path, "r") as f:
            lines = f.read().splitlines()
        return lines

    def loadDataset(self):
        self.dataset = KITTI
        dataPath = os.path.join(os.path.dirname(__file__), "data", "KITTI")
        filepath = os.path.join(dataPath, "splits", "eigen_zhou", "{}_files.txt")
        trainFilenames = self.readlines(filepath.format("train"))
        valFilenames = self.readlines(filepath.format("val"))
        numTrain = len(trainFilenames)
        self.numSteps = numTrain//(self.batchSize*self.epochs)
        trainDataset = self.dataset(dataPath, trainFilenames, self.height, self.width,
                                    self.frameIdxs, 4, True)
        valDataset = self.dataset(dataPath, valFilenames, self.height, self.width, self.frameIdxs,
                                  4, False)
        self.trainLoader = DataLoader(trainDataset, self.batchSize, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
        self.valLoader = DataLoader(valDataset, self.batchSize, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
        self.valIterator = iter(self.valLoader)

    def setTrain(self):
        for model in self.models.values():
            model.train()

    def setEval(self):
        for model in self.models.values():
            model.eval()

    def saveModel(self):
        outpath = os.path.join(os.path.dirname(__file__), "models", "weights_{}".format(self.epoch))
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        for name, model in self.models.items():
            savePath = os.path.join(outpath, "{}.pth".format(name))
            toSave = model.state_dict()
            if name == "encoder":
                toSave["height"] = self.height
                toSave["width"] = self.width
            torch.save(toSave, savePath)
        savePath = os.path.join(outpath, "adam.pth")
        torch.save(self.optimizer.state_dict(), savePath)

    def dispToDepth(self, disp, minDepth, maxDepth):
        minDisp = 1 / maxDepth
        maxDisp = 1 / minDepth
        scaledDisp = minDisp + (maxDisp - minDisp)*disp
        depth = 1 / scaledDisp
        return scaledDisp, depth

    def rotationFromAxisAngle(self, axisangle):
        angle = torch.norm(axisangle, 2, 2, True)
        axis = axisangle / (angle + 1e-7)
        cosAngle = torch.cos(angle)
        sinAngle = torch.sin(angle)
        complementCos = 1 - cosAngle
        x = axis[..., 0].unsqueeze(1)
        y = axis[..., 1].unsqueeze(1)
        z = axis[..., 2].unsqueeze(1)
        xs = x * sinAngle
        ys = y * sinAngle
        zs = z * sinAngle
        xcomplementCos = x * complementCos
        ycomplementCos = y * complementCos
        zcomplementCos = z * complementCos
        xycomplementCos = x * ycomplementCos
        yzcomplementCos = y * zcomplementCos
        zxcomplementCos = z * xcomplementCos
        rot = torch.zeros((axisangle.shape[0], 4, 4)).to(device=axisangle.device)
        rot[:, 0, 0] = torch.squeeze(x * xcomplementCos + cosAngle)
        rot[:, 0, 1] = torch.squeeze(xycomplementCos - zs)
        rot[:, 0, 2] = torch.squeeze(zxcomplementCos + ys)
        rot[:, 1, 0] = torch.squeeze(xycomplementCos + zs)
        rot[:, 1, 1] = torch.squeeze(y * ycomplementCos + cosAngle)
        rot[:, 1, 2] = torch.squeeze(yzcomplementCos - xs)
        rot[:, 2, 0] = torch.squeeze(zxcomplementCos - ys)
        rot[:, 2, 1] = torch.squeeze(yzcomplementCos + xs)
        rot[:, 2, 2] = torch.squeeze(z * zcomplementCos + cosAngle)
        rot[:, 3, 3] = 1
        return rot

    def getTranslationMatrix(self, translation):
        T = torch.zeros(translation.shape[0], 4, 4).to(device=translation.device)
        t = translation.contiguous().view(-1, 3, 1)
        T[:, 0, 0] = 1
        T[:, 1, 1] = 1
        T[:, 2, 2] = 1
        T[:, 3, 3] = 1
        T[:, :3, 3, None] = t
        return T

    def transformParameters(self, axisangle, translation, invert=False):
        rotation = self.rotationFromAxisAngle(axisangle)
        trans = translation.clone()
        if invert:
            rotation = rotation.transpose(1, 2)
            trans *= -1
        T = self.getTranslationMatrix(trans)
        if invert:
            M = torch.matmul(rotation, T)
        else:
            M = torch.matmul(T, rotation)
        return M

    def predictPoses(self, inputs, features):
        outputs = {}
        poseFeatures = {fi: inputs["color_aug", fi, 0] for fi in self.frameIdxs}
        for fi in self.frameIdxs[1:]:
            if fi < 0:
                poseInputs = [poseFeatures[fi], poseFeatures[0]]
            else:
                poseInputs = [poseFeatures[0], poseFeatures[fi]]
            poseInputs = [self.models["pose_encoder"](torch.cat(poseInputs, 1))]
            axisangle, translation = self.models["pose"](poseInputs)
            outputs[("axisangle", 0, fi)] = axisangle
            outputs[("translation", 0, fi)] = translation
            outputs[("cam_T_cam", 0, fi)] = self.transformParameters(axisangle[:, 0], translation[:, 0], invert=(fi<0))
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
                cameraPoints = self.backprojectDepth[sourceScale](depth, inputs[("inv_K", sourceScale)])
                pixelCoordinates = self.project3d[sourceScale](cameraPoints, inputs[("K", sourceScale)], T)
                outputs[("sample", frameIdx, scale)] = pixelCoordinates
                outputs[("color", frameIdx, scale)] = F.grid_sample(inputs[("color", frameIdx, sourceScale)],
                                                                    outputs[(("sample", frameIdx, scale))],
                                                                    padding_mode="border")

    def computeDepthErrors(self, depthGroundTruth, depthPred):
        threshold = torch.max((depthGroundTruth/depthPred), (depthPred/depthGroundTruth))
        a1 = (threshold < 1.25).float().mean()
        a2 = (threshold < 1.25**2).float().mean()
        a3 = (threshold < 1.25**3).float().mean()
        rootMeanSquaredError = (depthGroundTruth - depthPred)**2
        rootMeanSquaredError = torch.sqrt(rootMeanSquaredError.mean())
        rootMeanSquaredErrorLog = (torch.log(depthGroundTruth) - torch.log(depthPred))**2
        rootMeanSquaredErrorLog = torch.sqrt(rootMeanSquaredErrorLog.mean())
        absolute = torch.mean(torch.abs(depthGroundTruth - depthPred)/depthGroundTruth)
        squared = torch.mean(((depthGroundTruth - depthPred)**2)/depthGroundTruth)
        return absolute, squared, rootMeanSquaredError, rootMeanSquaredErrorLog, a1, a2, a3

    def computeDepthLosses(self, inputs, outputs, losses):
        depthPred = outputs[("depth", 0, 0)]
        depthPred = torch.clamp(F.interpolate(depthPred, [375, 1242], mode='bilinear',
                                              align_corners=False), 1e-3, 80)
        depthPred = depthPred.detach()
        depthGroundTruth = inputs["depth_gt"]
        mask = depthGroundTruth > 0
        cropMask = torch.zeros_like(mask)
        cropMask[:, :, 153:371, 44:1197] = 1
        mask = mask * cropMask
        depthGroundTruth = depthGroundTruth[mask]
        depthPred = depthPred[mask]
        depthPred *= torch.median(depthGroundTruth)/torch.median(depthPred)
        depthPred = torch.clamp(depthPred, 1e-3, 80)
        depthErrors = self.computeDepthErrors(depthGroundTruth, depthPred)
        for i, name in enumerate(self.depthMetricNames):
            losses[name] = np.array(depthErrors[i].cpu())

    def computeReprojectionLoss(self, pred, target):
        absDiff = torch.abs(pred - target)
        l1Loss = absDiff.mean(1, True)
        return l1Loss

    def getSmoothLoss(self, disp, img):
        gradientDispX = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        gradientDispY = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
        gradientImgX = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        gradientImgY = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
        gradientDispX *= torch.exp(-gradientImgX)
        gradientDispY *= torch.exp(-gradientImgY)
        return gradientDispX.mean() + gradientDispY.mean()

    def computeLosses(self, inputs, outputs):
        losses = {}
        totalLoss = 0
        for scale in range(self.numScales):
            loss = 0
            reprojectionLoss = []
            sourceScale = 0
            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, sourceScale)]
            for frameIdx in self.frameIdxs[1:]:
                pred = outputs[("color", frameIdx, scale)]
                reprojectionLoss.append(self.computeReprojectionLoss(pred, target))
            reprojectionLoss = torch.cat(reprojectionLoss, 1)
            combined = reprojectionLoss
            if combined.shape[1] == 1:
                toOptimise = combined
            else:
                toOptimise, idxs = torch.min(combined, dim=1)
            loss += toOptimise.mean()
            meanDisp = disp.mean(2, True).mean(3, True)
            normDisp = disp / (meanDisp + 1e-7)
            smoothLoss = self.getSmoothLoss(normDisp, color)
            loss += (1e-3 * smoothLoss)/(2**scale)
            totalLoss += loss
            losses["loss/{}".format(scale)] = loss
        totalLoss /= self.numScales
        losses["loss"] = totalLoss
        return losses

    def processBatch(self, inputs):
        for key, value in inputs.items():
            inputs[key] = value.to(self.device)
        features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs = self.models["decoder"](features)
        outputs.update(self.predictPoses(inputs, features))
        self.generateImagePredictions(inputs, outputs)
        losses = self.computeLosses(inputs, outputs)
        return outputs, losses

    def runEpoch(self):
        self.lrScheduler.step()
        self.setTrain()
        for batchIdx, inputs in enumerate(self.trainLoader):
            print("Epoch : {}, Batch : {}".format(self.epoch, batchIdx))
            outputs, losses = self.processBatch(inputs)
            self.optimizer.zero_grad()
            losses["loss"].backward()
            self.optimizer.step()
            self.computeDepthLosses(inputs, outputs, losses)
            self.val()

    def train(self):
        self.epoch = 0
        for self.epoch in range(self.epochs):
            print("Training --- Epoch : {}".format(self.epoch))
            self.runEpoch()
            if self.epoch % 5 == 0:
                self.saveModel()

    def val(self):
        self.setEval()
        try:
            inputs = self.valIterator.next()
        except:
            self.valIterator = iter(self.valLoader)
            inputs = self.valIterator.next()
        with torch.no_grad():
            outputs, losses = self.processBatch(inputs)
            self.computeDepthLosses(inputs, outputs, losses)
            del inputs, outputs, losses
        self.setTrain()

if __name__ == "__main__":
    t = Trainer()
    t.train()
