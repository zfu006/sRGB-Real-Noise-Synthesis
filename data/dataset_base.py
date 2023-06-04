from torch.utils.data import Dataset 
import torch
import os
import numpy as np
from abc import ABC, abstractmethod
import cv2
import glob

"""
# --------------------------------------------
# define a base dataset
# --------------------------------------------
"""
class dataset_base(Dataset, ABC):
    """This is an abstract class for datesets"""

    @abstractmethod
    def __init__(self):
        """Initialize the dataset, this should be overriden by the subclass"""
        pass

    @abstractmethod
    def __len__(self):
        """Return the number of images in the dataset, this should be overriden by the subclass"""

        pass

    @abstractmethod
    def __getitem__(self, index):
        """Return the image at index, this should be overriden by the subclass"""

        pass

    def _get_img_paths(self, root_dir, cam_name, flag='noisy'):
        sub_dirs = os.listdir(root_dir)
        img_lists = []

        for sub_dir in sub_dirs:
            if cam_name in sub_dir:
                img_dir = os.path.join(root_dir, sub_dir, flag, '*.png')     
                img_lists.extend(glob.glob(img_dir))

        return img_lists
    
    def _padding(self, img, win_size=10, factor=8):
        # pad the image if not_multiple of win_size * 8, win_size is the precision of padding stride
        multiple_factor = win_size*factor
        h, w, _ = img.shape
        H, W = (h//multiple_factor+1)*multiple_factor, (w//multiple_factor+1)*multiple_factor
        padh = H - h if h%multiple_factor != 0 else 0
        padw = W - w if w%multiple_factor != 0 else 0
        patch = cv2.copyMakeBorder(img, top=0, bottom=padh, left=0, right=padw, borderType=cv2.BORDER_REFLECT)
        return patch, h, w
    
    def _PD_scheme(self, img, stride=2):
        """input: img (tensor), output: img_mosaic (tensor)"""
        num = stride*stride
        idx = np.random.randint(0, num)
        h_start = idx // stride
        w_start = idx % stride
        img_mosaic = img[:, h_start::stride, w_start::stride]
        return img_mosaic