# Image Attention Transformer Network for Indoor 3D Object Detection




The main model is in ./model/detector.py
The evalution log is in ./eval_nolastconv_48g_4-0_lr/eval_log.txt

The best results are in line "eval INFO: T[2] IoU[0.25]" of evalution log 
and the AVG results are in line "eval INFO: AVG IoU[0.25]" of evalution log




## Install

### Requirements

- `Ubuntu 16.04`
- `Anaconda` with `python=3.6`
- `pytorch>=1.3`
- `torchvision` with  `pillow<7`
- `cuda=10.1`
- `trimesh>=2.35.39,<2.35.40`
- `'networkx>=2.2,<2.3'`
- compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413), which we used in the backbone
  network: `sh init.sh`
- others: `pip install matplotlib termcolor opencv-python tensorboard plyfile tqdm`


