```
mkdir /scratch/$USER/mondep_env
cd /scratch/$USER/mondep_env
cp -rp /scratch/work/public/overlay-fs-ext3/overlay-10GB-400K.ext3.gz .
gunzip overlay-10GB-400K.ext3.gz

singularity exec --overlay overlay-10GB-400K.ext3:rw /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh -b -p /ext3/miniconda3
```

create /ext3/env.sh and add:
#!/bin/bash

source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
export PYTHONPATH=/ext3/miniconda3/bin:$PATH


```
source /ext3/env.sh

conda update -n base conda -y
conda clean --all --yes
conda install pip
conda install ipykernel

mkdir -p ~/.local/share/jupyter/kernels
cd ~/.local/share/jupyter/kernels
cp -R /share/apps/mypy/src/kernel_template ./mondep_env
cd ./mondep_env 


```
then edit python wrapper file 
```
vi python
```
Adding
```
singularity exec $nv \
  --overlay /scratch/$USER/mondep/overlay-10GB-400K.ext3:rw \
  /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash -c "source /ext3/env.sh; $cmd $args"
```

Edit the default kernel.json file by setting PYTHON_LOCATION and KERNEL_DISPLAY_NAME using a text editor like nano/vim.

```
{
 "argv": [
  "/home/am11533/.local/share/jupyter/kernels/monodepth3_env/python",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "monodepth3_env",
 "language": "python"
}
```
