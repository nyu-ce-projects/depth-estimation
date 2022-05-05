import os
import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils import secondsToHM

from Models.EncoderModel import EncoderModel
from Models.DecoderModel import PoseDecoderModel
from Models.BackprojectDepth import BackprojectDepth
from Models.Project3D import Project3D
from Models.DisparityAdjustmentV2 import DisparityAdjustment
from Models.dpt.models import DPTDepthModel

from Losses.SSIM import SSIM

from Dataset.KITTI import KITTI

class Trainer:
    def __init__(self,config):
        self.config = config
        self.config['lr'] = float(config['lr'])
        self.batchSize = int(config['batchsize'])
        self.epochs = int(config['epochs'])
        self.height = int(config['height'])
        self.width = int(config['width'])
        self.frameIdxs = config['frame_ids']
        self.numScales = int(config['numscales'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.models = {}
        self.totalTrainableParams = 0
        self.trainableParameters = []

        # Depth Estimation Model Initialization
        self.models["depth"] = DPTDepthModel(backbone="vitl16_384",non_negative=True,enable_attention_hooks=False,)
        self.models["depth"] = self.models["depth"].to(self.device)
        self.trainableParameters += list(self.models["depth"].parameters())
        self.totalTrainableParams += sum(p.numel() for p in self.models["depth"].parameters() if p.requires_grad)
        
        # Pose Estimation Model Initialization
        self.models["encoder"] = EncoderModel(50)
        self.models["encoder"] = self.models["encoder"].to(self.device)
        self.trainableParameters += list(self.models["encoder"].parameters())
        self.totalTrainableParams += sum(p.numel() for p in self.models["encoder"].parameters() if p.requires_grad)

        self.models["pose"] = PoseDecoderModel(self.models["encoder"].numChannels, 2, 1)
        self.models["pose"] = self.models["pose"].to(self.device)
        self.trainableParameters += list(self.models["pose"].parameters())
        self.totalTrainableParams += sum(p.numel() for p in self.models["pose"].parameters() if p.requires_grad)
        
        #Disparity Adjustment Model
        self.disparityadjustment = DisparityAdjustment(self.device)
        
        # Loss, Metrics and Optimizer Init
        self.ssim = SSIM()
        self.ssim = self.ssim.to(self.device)
        print(self.config['lr'])
        self.optimizer = eval("optim."+self.config['optimizer'])(self.trainableParameters, lr=self.config['lr'], weight_decay=float(self.config['optim_weight_decay']))
        self.lrScheduler = eval("optim.lr_scheduler."+self.config['lr_scheduler'])(self.optimizer, int(self.config['lr_scheduler_steps']), float(self.config['lr_scheduler_decay']))
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
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.config['base_log_dir'],self.config['model_name'], mode))

    def readlines(self, path):
        with open(path, "r") as f:
            lines = f.read().splitlines()
        return lines

    def loadDataset(self):
        self.dataset = KITTI
        dataPath = self.config['datapath']
        filepath = os.path.join(dataPath, "splits", "eigen_zhou", "{}_files.txt")
        trainFilenames = self.readlines(filepath.format("train"))
        valFilenames = self.readlines(filepath.format("val"))
        numTrain = len(trainFilenames)
        self.numSteps = (numTrain//self.batchSize)*self.epochs
        trainDataset = self.dataset(dataPath, trainFilenames, self.height, self.width,
                                    self.frameIdxs, 4, True)
        valDataset = self.dataset(dataPath, valFilenames, self.height, self.width, self.frameIdxs,
                                  4, False)
        self.trainLoader = DataLoader(trainDataset, self.batchSize, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
        self.valLoader = DataLoader(valDataset, self.batchSize, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
        self.valIterator = iter(self.valLoader)

    def setTrain(self):
        for model in self.models.values():
            model.train()

    def setEval(self):
        for model in self.models.values():
            model.eval()
            
    def log(self, mode, inputs, outputs, losses):
        writer = self.writers[mode]
        for lossname, value in losses.items():
            writer.add_scalar("{}".format(lossname), value, self.step)
        for i in range(4):
            for frameIdx in self.frameIdxs:
                writer.add_image("color_{}/{}".format(frameIdx, i), inputs[("color", frameIdx, 0)][i].data, self.step)
                if frameIdx != 0:
                    writer.add_image("color_pred_{}/{}".format(frameIdx, i), outputs[("color", frameIdx, 0)][i].data, self.step)
                writer.add_image("disp/{}".format(i), self.normalizeImage(outputs[("disp", 0)][i]), self.step)
    
    def logTime(self, batchIdx, duration, loss):
        samplesPerSec = self.batchSize / duration
        totalTime = time.time() - self.startTime
        timeLeft = (self.numSteps / self.step - 1.0)*totalTime if self.step > 0 else 0
        logString = "Epoch : {:>3} | Batch : {:>7}, examples/s: {:5.1f}, loss : {:.5f}, time elapsed: {}, time left: {}"
        print(logString.format(self.epoch, batchIdx, samplesPerSec, loss, secondsToHM(totalTime), secondsToHM(timeLeft)))

    def saveModel(self):
        outpath = os.path.join(self.config['model_path'],self.config['model_name'], "weights_{}".format(self.epoch))
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
        
    def normalizeImage(self, image):
        maxValue = float(image.max().cpu().data)
        minValue = float(image.min().cpu().data)
        diff = (maxValue - minValue) if maxValue != minValue else 1e5
        return (image - minValue)/diff

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
        poseFeatures = {fi: features[fi] for fi in self.frameIdxs}
        for fi in self.frameIdxs[1:]:
            if fi < 0:
                poseInputs = [poseFeatures[fi], poseFeatures[0]]
            else:
                poseInputs = [poseFeatures[0], poseFeatures[fi]]
            axisangle, translation = self.models["pose"](poseInputs)
            outputs[("axisangle", 0, fi)] = axisangle
            outputs[("translation", 0, fi)] = translation
            outputs[("cam_T_cam", 0, fi)] = self.transformParameters(axisangle[:, 0], translation[:, 0], invert=(fi<0))
        return outputs

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
                T = outputs[("cam_T_cam", 0, frameIdx)]
                cameraPoints = self.backprojectDepth[sourceScale](depth, inputs[("inv_K", sourceScale)])
                pixelCoordinates = self.project3d[sourceScale](cameraPoints, inputs[("K", sourceScale)], T)
                outputs[("sample", frameIdx, scale)] = pixelCoordinates
                outputs[("color", frameIdx, scale)] = F.grid_sample(inputs[("color", frameIdx, sourceScale)],
                                                                    outputs[(("sample", frameIdx, scale))],
                                                                    padding_mode="border")
                outputs[("color_identity", frameIdx, scale)] = inputs[("color", frameIdx, sourceScale)]

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
        absDiff = torch.abs(target - pred)
        l1Loss = absDiff.mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        return 0.85*ssim_loss + 0.15*l1Loss

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
            identityReprojectionLoss = []
            for frameIdx in self.frameIdxs[1:]:
                pred = inputs[("color", frameIdx, sourceScale)]
                identityReprojectionLoss.append(self.computeReprojectionLoss(pred, target))
            identityReprojectionLoss = torch.cat(identityReprojectionLoss, 1)
            identityReprojectionLoss += torch.randn(identityReprojectionLoss.shape, device=self.device) * 0.00001
            combined = torch.cat((identityReprojectionLoss, reprojectionLoss), 1)
            if combined.shape[1] == 1:
                toOptimise = combined
            else:
                toOptimise, idxs = torch.min(combined, dim=1)
            outputs["identity_selection/{}".format(scale)] = (idxs > identityReprojectionLoss.shape[1] - 1).float()
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
        origScaleColorAug = torch.cat([inputs[("color_aug", fi, 0)] for fi in self.frameIdxs])
        allFrameFeatures = self.models["encoder"](origScaleColorAug)
        allFrameFeatures = [torch.split(f, self.batchSize) for f in allFrameFeatures]
        features = {}
        for i, frameIdx in enumerate(self.frameIdxs):
            features[frameIdx] = [f[i] for f in allFrameFeatures]
        outputs = self.models["depth"](features[0])
        outputs.update(self.predictPoses(inputs, features))
        self.generateImagePredictions(inputs, outputs)
        losses = self.computeLosses(inputs, outputs)
        return outputs, losses

    def runEpoch(self):
        self.setTrain()
        for batchIdx, inputs in enumerate(self.trainLoader):
            startTime = time.time()
            outputs, losses = self.processBatch(inputs)
            self.optimizer.zero_grad()
            losses["loss"].backward()
            self.optimizer.step()
            duration = time.time() - startTime
            early_phase = batchIdx % 200 == 0 and self.step < 2000
            late_phase = self.step % 1000 == 0
            if early_phase or late_phase:
                self.logTime(batchIdx, duration, losses["loss"].cpu().data)
                self.computeDepthLosses(inputs, outputs, losses)
                self.log("train", inputs, outputs, losses)
                self.val()
            self.step += 1
        self.lrScheduler.step()

    def train(self):
        print("Total Trainable Parameters : {}".format(self.totalTrainableParams))
        print("Total Steps : {}".format(self.numSteps))
        self.epoch = 0
        self.step = 0
        self.startTime = time.time()
        for self.epoch in range(self.epochs):
            print("Training --- Epoch : {}".format(self.epoch))
            self.runEpoch()
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
            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses
        self.setTrain()

if __name__ == "__main__":
    t = Trainer()
    t.train()
