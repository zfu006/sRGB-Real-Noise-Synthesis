import cv2 
import os
import numpy as np
import torch
import math
import random
import glob


'''
# --------------------------------------------
# read image from path
# opencv is fast, but read BGR numpy image
# --------------------------------------------
'''


# --------------------------------------------
# get uint8 image of size HxWxn_channles (RGB)
# --------------------------------------------
def imread_uint(path, n_channels=3):
    # input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img
    
# --------------------------------------------
# matlab's imwrite
# --------------------------------------------
def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)



"""
# ----------------------------------------
# data transform 
# ----------------------------------------
"""

def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis = 2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.)
   
def tensor2uint(img):
    img = img.data.squeeze().float().clamp(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255).round())

def uint2single(img):

    return np.float32(img/255.)


def single2uint(img):

    return np.uint8((img.clip(0, 1)*255.).round())

def single2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis = 2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
   
def tensor2single(img):
    img = img.data.squeeze().float().clamp(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return img

def tensor2data(img):
    img = img.data.squeeze().float().cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return img

"""
# ----------------------------------------------
# Data augmentation
# ----------------------------------------------
"""

def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

class pixel_shuffle():
    def __init__(self, stride=2):
        num = stride ** 2
        self.idx = np.random.randint(0, num)
        self.stride = stride
        self.h_start = self.idx // stride
        self.w_start = self.idx % stride

    def _crop(self, img):
        img_mosaic = img[:, :, self.h_start::self.stride, self.w_start::self.stride]
        return img_mosaic

'''
# ----------------------------------------------
# metric, PSNR, SSIM, Discrete KL Divergence...
# ----------------------------------------------
'''
# --------------------------------------------
# PSNR
# --------------------------------------------
def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

# ----------------------------------------------------
# KL Divergence
# ----------------------------------------------------
def noise_quantization(noisy, clean):
    """Discretize the siythesied noise"""
    noisy = torch.clip(noisy, 0, 1)
    noisy = torch.round(noisy * 255) / 255.0
    noise = noisy - clean
    return noise

def get_histogram(data, bin_edges=None, left_edge=0.0, right_edge=1.0, n_bins=1000):
    data_range = right_edge - left_edge
    bin_width = data_range / n_bins
    if bin_edges is None:
        bin_edges = np.arange(left_edge, right_edge + bin_width, bin_width)
    bin_centers = bin_edges[:-1] + (bin_width / 2.0)
    n = np.prod(data.shape)
    hist, _ = np.histogram(data, bin_edges)
    return hist / n, bin_centers

def cal_kld(p_data, q_data, left_edge=0.0, right_edge=1.0, n_bins=1000):
    """Returns forward, inverse, and symmetric KL divergence between two sets of data points p and q"""
    bw = 0.2 / 64
    bin_edges = np.concatenate(([-1000.0], np.arange(-0.1, 0.1 + 1e-9, bw), [1000.0]), axis=0)
    p, _ = get_histogram(p_data, bin_edges, left_edge, right_edge, n_bins) 
    q, _ = get_histogram(q_data, bin_edges, left_edge, right_edge, n_bins)
    idx = (p > 0) & (q > 0)
    p = p[idx]
    q = q[idx]
    logp = np.log(p)
    logq = np.log(q)
    kl_fwd = np.sum(p * (logp - logq))
    kl_inv = np.sum(q * (logq - logp))
    kl_sym = (kl_fwd + kl_inv) / 2.0
    return kl_fwd #, kl_inv, kl_sym

"""
# -------------------------------------
# Crop Image
# -------------------------------------
# """

def crop_sidd_dataset(ori_dir, target_dir='./Datasets/SIDD_Medium_Cropped/', p_size=512, p_overlap=128, p_max=800):
    """crop images for SIDD dataset
        
        Args:
        ori_dir -- the root of original SIDD dataset
        target_dir -- the root of cropped SIDD dataset
        p_size -- patch size
        p_overlap -- the overlap of two patches
        p_max -- the minimum size of images to be cropped

    """
    subdirs = os.listdir(ori_dir)
    for subdir in subdirs:
        for full_path, _, file_names in os.walk(os.path.join(ori_dir, subdir)):
            for file_name in file_names:
                img_name = os.path.splitext(file_name)[0]
                img = imread_uint(os.path.join(full_path, file_name), n_channels=3)
                if 'GT' in file_name:
                    target_path = os.path.join(target_dir, subdir, 'clean')
                elif 'NOISY' in file_name:
                    target_path = os.path.join(target_dir, subdir, 'noisy')
                else:
                    raise ValueError('Wrong image name.')
                if not os.path.exists(target_path):
                    os.makedirs(target_path)
                h, w = img.shape[:2]
                if h > p_max and w > p_max:
                    num = 0
                    w1 = list(np.arange(0, w-p_size, p_size-p_overlap, dtype=np.int))
                    h1 = list(np.arange(0, h-p_size, p_size-p_overlap, dtype=np.int))
                    w1.append(w-p_size)
                    h1.append(h-p_size)
                    for i in h1:
                        for j in w1:
                            num += 1
                            patch = img[i:i+p_size, j:j+p_size, :]
                            imsave(patch, os.path.join(target_path, img_name + '_{:05d}.png'.format(num)))
                else:
                    imsave(img, os.path.join(target_path, img_name, '.png'))
                

def train_test_split(ori_dir, target_dir, cam_name, ratio=0.8):
    train_file_name = cam_name + '_train.txt'
    test_file_name = cam_name + '_test.txt'
    train_file_path = os.path.join(target_dir, train_file_name)
    test_file_path = os.path.join(target_dir, test_file_name)
    train_file = open(train_file_path, 'w')
    test_file = open(test_file_path, 'w')

    subdirs = os.listdir(ori_dir)
    selected_dirs = [dir_name for dir_name in subdirs if cam_name in dir_name and os.path.isdir(os.path.join(ori_dir, dir_name))]
    classify_dirs = {}
    for selected_dir in selected_dirs:
        ISO_level = selected_dir[12:17] # split the ISO level from the SIDD dataset
        if ISO_level not in classify_dirs.keys():
            classify_dirs[ISO_level] = []
        classify_dirs[ISO_level].append(selected_dir)

    for ISO_level in classify_dirs.keys():
        dirs = classify_dirs[ISO_level]
        num_dirs = len(dirs)
        num_train = round(num_dirs*ratio)
        num_test = num_dirs - num_train
        train_dirs = random.sample(dirs, num_train)
        test_dirs = list(set(dirs) - set(train_dirs))
        
        for train_dir in train_dirs:
            train_file.write(train_dir + '\n')
        
        for test_dir in test_dirs:
            test_file.write(test_dir + '\n')

    train_file.close()
    test_file.close()

def get_img_list(ori_dir, cam_name, mode='train', ratio=0.9):
    """get the image list for training and testing. 
       Training and validation sets are randomly selected from the training set."""
    if mode == 'train':
        file_name = cam_name + '_train.txt'
        file = open('./data_preparation/' + file_name, 'r')
        sub_dirs = file.readlines()
        file.close()
        img_lists = []
        for sub_dir in sub_dirs:
            full_dir = os.path.join(ori_dir, sub_dir.strip(), 'noisy', '*.png') # noisy images
            img_lists.extend(glob.glob(full_dir))
        num_train = round(len(img_lists)*ratio)
        train_lists = random.sample(img_lists, num_train)
        val_lists = list(set(img_lists) - set(train_lists))

        return train_lists, val_lists
    
    elif mode == 'test':
        file_name = cam_name + '_test.txt'
        file = open('./data_preparation/' + file_name, 'r')
        sub_dirs = file.readlines()
        file.close()
        test_lists = []
        for sub_dir in sub_dirs:
            full_dir = os.path.join(ori_dir, sub_dir.strip(), 'noisy', '*.png')
            test_lists.extend(glob.glob(full_dir))
        return test_lists

    else:
        raise ValueError('Wrong mode.')

        



if __name__ == '__main__':
    ori_path = './Datasets/SIDD_Medium_Crop/'
    target_path = './data_preparation/'
    train_test_split(ori_path, target_path, 'G4', 0.8)