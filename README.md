# Voxel Mamba: Group-Free State Space Models for Point Cloud based 3D Object Detection

This repo is the official implementation of the paper [Voxel Mamba: Group-Free State Space Models for Point Cloud based 3D Object Detection](). Our Voxel Mamba achieves state-of-the-art performance on Waymo and nuScene datasets.  

## News
-[24-6-14] Voxel Mamba will be released on [arxiv]()

## TODO
- [ ] Release the [arXiv]() version.
- [ ] Clean up and release the code.
- [ ] Release code of Waymo.
- [ ] Release code of NuScenes.
- [ ] Merge Voxel Mamba to [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

## Main Results
#### Waymo Open Dataset
Validation set  
|  Model  | mAPH_L1 | mAPH_L2 | Veh_L1 | Veh_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 | Log |
|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
|  [Voxel Mamba]() | 79.6  |  73.6  | 80.8/80.3 | 72.6/72.2 | 85.0/80.8 | 77.7/73.6 | 78.6/77.6 | 75.7/74.8 | [Log]() | 

Test set
|  Model  | mAPH_L1 | mAPH_L2 | Veh_L1 | Veh_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 | Log |
|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
|  [Voxel Mamba]() | 79.6  |  74.3  | 84.4/84.0 | 77.0/76.6 | 84.8/80.6 | 79.0/74.9 | 75.4/74.3 | 72.6/71.5 | [Log]() | 


#### nuScene Dataset
Validation set  
|  Model  | mAP | NDS | mATE | mASE | mAOE | mAVE| mAAE | ckpt | Log |
|---------|---------|--------|---------|---------|--------|---------|--------|--------|--------|
|  [Voxel Mamba]() | 67.5 | 71.9 | 26.7 | 25.0 | 25.8 | 21.8 | 18.9| [ckpt]()| [Log]()|  

Test set  
|  Model  | mAP | NDS | mATE | mASE | mAOE | mAVE| mAAE | ckpt | Log |
|---------|---------|--------|---------|---------|--------|---------|--------|--------|--------|
|  [Voxel Mamba]() | 69.0 | 73.0 | 24.3 | 23.0 | 30.9 | 23.7 | 13.3| [ckpt]()| [Log]()|  


Voxel Mamba's result on Waymo compared with other leading methods.
All the experiments are evaluated on an NVIDIA A100 GPU with the same environment.
We hope that our Voxel Mamba can provide a potential group-free solution for efficiently handling sparse point clouds for 3D tasks.
<div align="left">
  <img src="docs/Speed_Performance.png" width="500"/>
</div>

## Usage
### Installation
Please refer to [INSTALL.md](docs/INSTALL.md) for installation.

### Dataset Preparation
Please follow the instructions from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md). We adopt the same data generation process.

### Generate Hilbert Template
```
cd data
mkdir hilbert
python ./tools/hilbert_curves/create_hilbert_curve_template.py
```


### Training
```
# multi-gpu training
cd tools
bash scripts/dist_train.sh 8 --cfg_file <CONFIG_FILE>
```

### Testing
```
# multi-gpu testing
cd tools
bash scripts/dist_test.sh 8 --cfg_file <CONFIG_FILE> --ckpt <CHECKPOINT_FILE>
```

## Citation
Please consider citing our work as follows if it is helpful.
```
```

## Acknowledgments
Voxel Mamba is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and [DSVT](https://github.com/Haiyang-W/DSVT).  
We also thank the Centerpoint, TransFusion, OctFormer, and HEDNet authors for their efforts.



