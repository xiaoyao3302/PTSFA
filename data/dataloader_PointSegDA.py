# PointSegDA dataset
import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from utils.pc_utils_Norm import (jitter_pointcloud, random_rotate_one_axis)
from augmentation import density, drop_hole, p_scan



class ReadSegData(Dataset):
    """
    modelnet dataset for pytorch dataloader
    """
    def __init__(self, io, dataroot, dataset, partition='train', random_rotation=True):
        # train, val, test
        self.partition = partition
        self.random_rotation = random_rotation

        self.data = []
        folders = os.path.join(dataroot, dataset, partition, '*.npy')
        data_files = glob.glob(folders)
        for file in data_files:
            self.data.append(np.load(file))

        self.num_examples = len(self.data)

        io.cprint("The number of " + partition + " examples in the " + dataset + " dataset is: " + str(self.num_examples))

    def __getitem__(self, item):
        pointcloud = self.data[item].astype(np.float32)[:, :3]
        label = self.data[item].astype(np.long)[:, 3] - 1  # labels 1-8

        # apply data rotation and augmentation on train samples
        if self.partition == 'train':
            pointcloud = jitter_pointcloud(pointcloud)
            if self.random_rotation == True:
                pointcloud = random_rotate_one_axis(pointcloud, "z")

        if self.partition == 'train':
            pointcloud_aug = pointcloud
            if np.random.random() > 0.5:
                pointcloud_aug = density(pointcloud_aug, num_point=2048)
            if np.random.random() > 0.5:
                pointcloud_aug = drop_hole(pointcloud_aug, num_point=2048)
            if np.random.random() > 0.5:
                pointcloud_aug = p_scan(pointcloud_aug, num_point=2048)
        else:
            pointcloud_aug = pointcloud
                
        return (item, pointcloud, label, pointcloud_aug)

    def __len__(self):
        return len(self.data)
    
