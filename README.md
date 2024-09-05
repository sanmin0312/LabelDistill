# LabelDistill: Label-guided Cross-modal Knowledge Distillation for Camera-based 3D Object Detection


<img src="figs/Video_daytime_scene.gif" width="1000" title="daytime scene" height="">

> [**LabelDistill: Label-guided Cross-modal Knowledge Distillation for Camera-based 3D Object Detection**](https://arxiv.org/abs/2407.10164)  
> [Sanmin Kim](https://scholar.google.co.kr/citations?user=CiMsEwgAAAAJ&hl=ko),
> Youngseok Kim, Sihwan Hwang, Hyeonjun Jeong, and Dongsuk Kum,
> [*ECCV 2024*](https://eccv2024.ecva.net/)

### Abstract
Recent advancements in camera-based 3D object detection have introduced cross-modal knowledge distillation to bridge the performance gap with LiDAR 3D detectors, leveraging the precise geometric information in LiDAR point clouds. 
However, existing cross-modal knowledge distillation methods tend to overlook the inherent imperfections of LiDAR, such as the ambiguity of measurements on distant or occluded objects, which should not be transferred to the image detector.
To mitigate these imperfections in LiDAR teacher, we propose a novel method that leverages aleatoric uncertainty-free features from ground truth labels.
In contrast to conventional label guidance approaches, we approximate the inverse function of the teacher's head to effectively embed label inputs into feature space.
This approach provides additional accurate guidance alongside LiDAR teacher, thereby boosting the performance of the image detector.
Additionally, we introduce feature partitioning, which effectively transfers knowledge from the teacher modality while preserving the distinctive features of the student, thereby maximizing the potential of both modalities.
Experimental results demonstrate that our approach improves mAP and NDS by 5.1 points and 4.9 points compared to the baseline model, proving the effectiveness of our approach.


## Getting Started

### Installation
**1. Create a conda virtual environment**
```
conda create --name labeldistill python=3.8 -y
conda activate labeldistill
```

**2. Install PyTorch (v1.9.0)**
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

**3. Clone repo**
```
git clone https://github.com/sanmin0312/LabelDistill.git
```

**4. Install mmcv, mmdet and mmseg**
```
pip install openmim
mim install mmcv-full==1.6.0
mim install mmdet==2.26.0
mim install mmsegmentation==0.29.1
```

**5. Install mmdet3d**
```
cd LabelDistill
git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
pip install -e .
```

**6. Install requirements**
```
cd ..
pip install -r requirements.txt
python setup.py develop
```


### Data Preparation

**1. Downlaod nuScenes official dataset & make symlink**
```
ln -s [nuscenes root] ./data/
```

**2. Prepare infos**
```
python scripts/gen_info.py
```

**3. Generate lidar depth**
```
python scripts/gen_depth_gt.py
```

The directory should be as follows.
```
LabelDistill
├── data
│   ├── nuScenes
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── maps
│   │   ├── samples
│   │   ├── depth_gt
│   │   ├── sweeps
│   │   ├── v1.0-trainval
```

# Training and Evaluation

**Training**
```
python [EXP_PATH] --amp_backend native -b 4 --gpus 4
```

**Evaluation**
```
python [EXP_PATH] --ckpt_path [CKPT_PATH] -e -b 4 --gpus 4
```

# Model Zoo
| Model | Backbone | Weight | Config | mAP | NDS |
| - | - | - | - | - | - |
| CenterPoint (LiDAR Teacher) | - | [link](https://drive.google.com/file/d/1YWasvUGLQyI0FtruVsMmzQ2TNtBNV7Ad/view?usp=drive_link) | [config](labeldistill/exps/nuscenes/labeldistill/centerpoint_vox01_128x128_20e_10sweeps.py) | 58.4 | 65.2 |
| Label Encoder (Label Teacher) | - | [link](https://drive.google.com/file/d/1FAzc2RAZQNM3dyv-nNbpWNkxHyGt1ujc/view?usp=drive_link) | [config](labeldistill/exps/nuscenes/labeldistill/LabelDistill_step1.py) | - | - |
| LabelDistill (Student) | ResNet-50 | [link](https://drive.google.com/file/d/1O-pTtZhcx0ZQX733QDjY9eHM0BAXm6dU/view?usp=drive_link) | [config](labeldistill/exps/nuscenes/labeldistill/LabelDistill_r50_128x128_e24_4key.py) | 41.9 | 52.8 |
# Citation
```
@inproceedings{kim2024labeldistill,
  title={LabelDistill: Label-guided Cross-modal Knowledge Distillation for Camera-based 3D Object Detection},
  author={Kim, Sanmin and Kim, Youngseok and Hwang, Sihwan and Jeong, Hyeonjun and Kum, Dongsuk},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2024}
}
```
# Acknowledgement
- [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
- [CenterPoint](https://github.com/open-mmlab/mmdetection3d/tree/main/configs/centerpoint)