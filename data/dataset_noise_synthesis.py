import os
import random
import utils.utils_image as util
from data.dataset_base import dataset_base

class dataset_noise_synthesis(dataset_base):
    def __init__(self,
                 img_lists,
                 img_channels,
                 not_train=False,
                 not_aug=False,
                 patch_size=96):
        
        self.noisy_img_lists = img_lists
        self.img_channels = img_channels
        self.not_train = not_train
        self.not_aug = not_aug
        self.patch_size = patch_size
    
    def __getitem__(self, index):
        noisy_img_path = self.noisy_img_lists[index]
        noisy_img_dir, noisy_img_name = os.path.split(noisy_img_path)
        clean_img_dir = noisy_img_dir.replace('noisy', 'clean')
        clean_img_name = noisy_img_name.replace('NOISY', 'GT')
        clean_img_path = os.path.join(clean_img_dir, clean_img_name)
        clean_img = util.imread_uint(clean_img_path, self.img_channels)
        noisy_img = util.imread_uint(noisy_img_path, self.img_channels)
        
        if not self.not_train:
            h, w, _ = clean_img.shape
            rnd_h = random.randint(0, max(0, h - self.patch_size))
            rnd_w = random.randint(0, max(0, w - self.patch_size))
            clean_patch = clean_img[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            noisy_patch = noisy_img[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            
            if not self.not_aug:
                mode = random.randint(0, 7)
                clean_patch = util.augment_img(clean_patch, mode)
                noisy_patch = util.augment_img(noisy_patch, mode)

                        
            clean_patch = util.uint2tensor3(clean_patch)
            noisy_patch = util.uint2tensor3(noisy_patch)

        else:
            clean_patch = util.uint2tensor3(clean_img)
            noisy_patch = util.uint2tensor3(noisy_img)

        noise_map = noisy_patch - clean_patch
        return {'clean': clean_patch, 'noisy': noisy_patch, 'noise': noise_map}
    
    def __len__(self):
        return len(self.noisy_img_lists) 
            