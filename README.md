# Monocular Depth Estimation using Self-Supervised Learning 

* [Environment Setup](#env)



# Environment Setup

### Using Conda 
```
conda env create -f depthestimate_env.yaml
conda activate depthestimate_env
```
for mac m1 use depthestimate_env_mac_cpu.yaml


Training your model
```
python train.py --model MONODEPTH2 --conf configs/model_config.cfg 
```

To run in background

```
nohup python -u train.py --model MONODEPTH2 > output.log &
```
| Model                        | Additions                                                                  | Link                                                                                                              |
|------------------------------|----------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| CAMLESS                      | Learnable Camera Intrinsics                                                | [ `link` ]( https://storage.googleapis.com/depth-estimation-weights/final_weights/CAMLESS.zip )                   |
| ESPCN                        | Using ESPCN for Upsampling                                                 | [ `link` ]( https://storage.googleapis.com/depth-estimation-weights/final_weights/ESPCN.zip )                     |
| CAMLESS_WEATHER_AUGMENTATION | CAMLESS with weather augmentation                                          | [ `link` ]( https://storage.googleapis.com/depth-estimation-weights/final_weights/CAMLESS_WEATHER_AUG.zip )       |
| MASKCAMLESS                  | Semantic segmentation suggestion from pretrained MASK-RCNN Model + CAMLESS | [ `link` ]( https://storage.googleapis.com/depth-estimation-weights/final_weights/MASKCAMLESS.zip )               |
| MASKCAMLESS_V2               | MASKCAMLESS + skipping loss adjustment for Smoothness loss                 | [ `link` ]( https://storage.googleapis.com/depth-estimation-weights/final_weights/MASKCAMLESS_V2.zip )            |
| MASKCAMLESS_ESPCN            | Mask R-CNN + CAMLESS + ESPCN                                               | [`link`](https://storage.googleapis.com/depth-estimation-weights/final_weights/MASKCAMLESS_ESPCN.zip)             |
| MASKCAMLESS_ESPCN_WEATHER    | MASKCAMLESS_ESPCN + weather augmentation                                   | [ `link` ]( https://storage.googleapis.com/depth-estimation-weights/final_weights/MASKCAMLESS_ESPCN_WEATHER.zip ) |
| MASKCAMLESS_ESPCN_V2         | MASKCAMLESS_ESPCN+ skipping loss adjustment for Smoothness loss            | [ `link` ]( https://storage.googleapis.com/depth-estimation-weights/final_weights/MASKCAMLESS_ESPCN_V2.zip )      |
