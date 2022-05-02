# MonDep - Monocular Depth Estimation using Self-Supervised Learning 

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
