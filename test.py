import os
import torch
import PIL.Image as pil
from torchvision import transforms

from Trainers.Trainer import getTrainer

from configs.config_loader import load_config
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np

def getImageInTensor(imgPath):
    img = pil.open(imgPath).convert('RGB')
    original_width, original_height = img.size
    img = img.resize((width, height), pil.LANCZOS)
    imgTensor = transforms.ToTensor()(img).unsqueeze(0)
    imgTensor = imgTensor.to(device)
    return imgTensor,original_width,original_height

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'MASKCAMLESS_ESPCN_WEATHER'
    weight_name = 'weights_10'
    path = os.path.join("trained_models",model_name, weight_name)

    config = load_config(config_path='configs/model_config.cfg',model_name=model_name)
    net = getTrainer(config)
    encoderDict = torch.load(os.path.join(path, "encoder.pth"), map_location=device)
    height = encoderDict.pop("height")
    width = encoderDict.pop("width")
    net.models["encoder"].load_state_dict(encoderDict)

    net.models["decoder"].load_state_dict(torch.load(os.path.join(path, "decoder.pth"), map_location=device))

    net.setEval()
    
    imgTensor,original_width,original_height = getImageInTensor(imgPath = "./external_img/snow_image.jpg")

    with torch.no_grad():
        features = net.models["encoder"](imgTensor)
        outputs = net.models["decoder"](features)
        disp = net.disparityadjustment(imgTensor,outputs[("disp", 0)])
        disp_resized = torch.nn.functional.interpolate(
                        disp, (original_height, original_width), mode="bilinear", align_corners=False)
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)

    im.save('./test.jpeg')