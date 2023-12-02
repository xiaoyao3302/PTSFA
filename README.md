# PTSFA

This is the official PyTorch implementation of our paper:

> **[Progressive Target-Styled Feature Augmentation for Unsupervised Domain Adaptation on Point Clouds](https://arxiv.org/abs/2311.16474)**


> **Abstract.** 
> Unsupervised domain adaptation is a critical challenge in the field of point cloud analysis, as models trained on one set of data often struggle to perform well in new scenarios due to domain shifts. Previous works tackle the problem by using adversarial training or self-supervised learning for feature extractor adaptation, but ensuring that features extracted from the target domain can be distinguished by the source-supervised classifier remains challenging. In this work, we propose a novel approach called progressive target-styled feature augmentation (PTSFA). Unlike previous works that focus on feature extractor adaptation, our PTSFA approach focuses on classifier adaptation. It aims to empower the classifier to recognize target-styled source features and progressively adapt to the target domain. To enhance the reliability of predictions within the PTSFA framework and encourage discriminative feature extraction, we further introduce a new intermediate domain approaching (IDA) strategy. We validate our method on the benchmark datasets, where our method achieves new state-of-the-art performance.



## Getting Started

### Installation

Please follow our requirements.txt to prepare the environment.



### Dataset:

Our code supports PointDA-10 dataset, GraspNetPC-10 dataset, and PointSegDA dataset.

- Please download PointDA-10 dataset at https://drive.google.com/file/d/1-LfJWL5geF9h0Z2QpdTL0n4lShy8wy2J/view?usp=sharing.

- Please download GraspNetPC-10 dataset at https://drive.google.com/file/d/1VVHmsSToFMVccge-LsYJW67IS94rNxWR/view?usp=sharing.
- Please download PointSegDA dataset at https://drive.google.com/uc?id=1dkU-8Y8K7yZaZjwelVUsxAbBf7JmOX9j.

Please unzip the datasets and modify the dataset path in configuration files.

For example, if you put your data under the data folder like this, you can directly bash our run_test.sh file to run the code.
```

├── data
    ├── GraspNetPointClouds
        ├── test
        └── train
    ├── PointDA_data
        ├── modelnet
        ├── scannet
        └── shapenet
    └── PointSegDAdataset
        ├── adobe
        ├── faust
        ├── mit
        └── scape
├── PTSFA
    ├── data
        ├── dataloader_XXXX.py
        ├── ....
        └── grasp_datautils
  	├── log
  		├── XXX.txt
  		└── XXX.txt
    ├── models 
        ├── model.py
        └── pointnet_util.py
    ├── utils
        ├── log_SPST.py
        ├── ....
        └── trans_norm.py
    ├── augmentation.py
    ├── ....
    └── train_GTSA_seg.py

```



## Usage

We tried many different methods before finally coming up with our PTSFA. So, we also provide the codes of some of the methods we tried, so the config may seem a bit cumbersome. 
If you want to reproduce our results, please directly bash run_test.sh.

To run with different settings, please modify the settings in the sh file.

We have uploaded the log files in the log folder.

Note that all of our experiments are tested on 4 2080Ti GPUs, on 2 A5000 GPUs or on 10 3090 GPUs.

P.S.

​	the current version is a bit complex, please allow me a couple of days to simply it (maybe longer), if you need.



## Citation

If you find these projects useful, please consider citing our paper.
```
@article{wang2023progressive,
  title={Progressive Target-Styled Feature Augmentation for Unsupervised Domain Adaptation on Point Clouds},
  author={Wang, Zicheng and Zhao, Zhen and Wu, Yiming and Zhou, Luping and Xu, Dong},
  journal={arXiv preprint arXiv:2311.16474},
  year={2023}
}
```



## Acknowledgement

We thank [GAST](https://github.com/zou-longkun/GAST), [MLSP](https://github.com/VITA-Group/MLSP), [DefRec_and_PCM](https://github.com/IdanAchituve/DefRec_and_PCM), [PointDAN](https://github.com/canqin001/PointDAN), [ImplicitPCDA](https://github.com/Jhonve/ImplicitPCDA), [DGCNN](https://github.com/WangYueFt/dgcnn), [PointNet](https://github.com/charlesq34/pointnet), [PointNet++](https://github.com/charlesq34/pointnet2) and other relevant works for their amazing open-sourced projects!
