# MonDep - Monocular Depth Estimation using Self-Supervised Learning 

* [Environment Setup](#env)



# Environment Setup

### Using Conda 
```
conda env create -f mondep_env.yaml
conda activate mondep_env
```


Training your model
```
python train.py --model MONODEPTH2 --conf configs/model_config.cfg 
```
