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



|Implementation                                                  |a1    |a2    |a3    |abs_rel|log_rms|rms  |sq_rel|
|----------------------------------------------------------------|------|------|------|-------|-------|-----|------|
|MonoDepth2 [6]                                                  |0.877 |0.959 |0.981 |0.115  |0.193  |4.863|0.903 |
|CamLess[10]                                                     |0.891 |0.964 |0.983 |0.106  |0.182  |4.482|0.75  |
|Ours - Monodepth2 +  Mask R-CNN                                 |0.9008|0.9684|0.9872|0.1117 |0.1886 |3.977|0.5114|
|Ours - MonoDepth2 + Mask R-CNN + ESPCN                          |0.8403|0.9651|0.9858|0.1214 |0.205  |4.096|0.6251|
|Ours - MonoDepth2 + CamLess                                     |0.8629|0.9542|0.98  |0.1186 |0.2103 |4.737|0.7843|
|Ours - MonoDepth2 + CamLess+Weather Augmentation                |0.8704|0.9582|0.9789|0.1223 |0.2016 |4.934|0.9271|
|Ours - MonoDepth2 + Mask R-CNN + CamLess                        |0.9148|0.9685|0.9832|0.0996 |0.1887 |4.25 |0.5722|
|Ours - MonoDepth2 + Mask R-CNN + CamLess (Adjusted Loss)        |0.879 |0.9699|0.9876|0.111  |0.177  |3.959|0.5079|
|Ours - MonoDepth2 + Mask R-CNN + ESPCN + CamLess                |0.9105|0.9637|0.9814|0.0956 |0.1858 |3.746|0.4868|
|Ours - MonoDepth2 + Mask R-CNN + ESPCN + CamLess (Adjusted Loss)|0.8854|0.9621|0.9842|0.1166 |0.1884 |3.485|0.4793|
