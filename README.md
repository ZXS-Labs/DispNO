# Disparity Map Neural Operator

## How to use

### Environment

- Python 3.6
- Pytorch 1.2

### Install

#### Create a virtual environment and activate it.

```
conda create -n DispNO python=3.6
conda activate DispNO
```

#### Dependencies

```
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
pip install opencv-python
pip install tensorboardx
pip install scikit-image
```

#### install deformable conv

```
cd deform_conv
python setup.py install
```

### Data Preparation

Download [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo), [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo), [Middlebury](https://vision.middlebury.edu/stereo/data/)

Synthetic dataset UnrealStereo4K is created by [Fabio Tosi](https://vision.disi.unibo.it/~ftosi/), refer to Link [SMD-Nets](https://github.com/fabiotosi92/SMD-Nets) for the method of downloading this dataset.

### Train

If you want to modify the configuration, please modify the files under the folder scripts.

```
sh scripts/sceneflowDispNO.sh
```

### Test

```
sh scripts/testDispNO.sh
```

### Fine-tune

For example fine-tuning on Middlebury, you can customize the scale range and disparity range in this file.

```
sh scripts/middlebury_DispNO_range.sh
```

#### Pretrained Model

epoch:50 

[Scene Flow](https://drive.google.com/drive/folders/1iZw0XkA2AjMOY5D2xLKghNYgyqn8r1oF?usp=sharing)

## Acknowledgement

We thank the authors that shared the code of their works. In particular:

- Xiaoyang Guo for providing the code of [GwcNet](https://github.com/xy-guo/GwcNet)
- Qi Zhang for providing the code of [HDA-Net](https://dl.acm.org/doi/abs/10.1145/3474085.3475273)
- Fabio Tosi for providing the code of [SMD-Nets](https://github.com/fabiotosi92/SMD-Nets)
- Min Wei for providing the code of [Super-Resolution-Neural-Operator](https://github.com/2y7c3/Super-Resolution-Neural-Operator)

Our work is inspired by these work.