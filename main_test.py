import torch
import torch.nn as nn
import argparse
import os
import numpy as np

from models.network_unet import UNet
from models.network_plain import Noise_estimator, Noise_level_predictor, DnCNN
import utils.utils_image as util

def main():

    opt = argparse.ArgumentParser() 
    opt.add_argument('--cam_name', type=str, default='IP', help='camera name for training data, the dataset will load the images from the specified camera')
    opt.add_argument('--dir_test_save', type=str, default='./test/saves/', help='save directory for test')
    opt.add_argument('--dir_test_img', type=str, default='./test/imgs/', help='image directory for test data')
    opt.add_argument('--NeCA_type', type=str, default='W', help='set mode of NeCA')
    opt.add_argument('--noise_level', type=int, default=25, help='set Gaussian noise level for NeCA_S')
    opt = opt.parse_args()

    net_E = Noise_estimator(in_nc=3, out_nc=1, nc=96, nb=5, act_mode='R')
    net_G1 = Noise_level_predictor(in_nc=3, out_nc=3, nc=96)
    net_G2 = UNet(in_nc=3, out_nc=3, nc=64, act_mode='R', num_stages=4, downsample_mode='strideconv', upsample_mode='upsampling', bias=True, padding_mode='zeros', final_act='Tanh')

    checkpoint_E = torch.load(os.path.join('./pretrain/', opt.cam_name, 'checkpoint_E.pth'), map_location='cpu')
    checkpoint_G1 = torch.load(os.path.join('./pretrain/', opt.cam_name, 'checkpoint_G1.pth'), map_location='cpu')
    checkpoint_G2 = torch.load(os.path.join('./pretrain/', opt.cam_name, 'checkpoint_G2.pth'), map_location='cpu')
    
    net_E.load_state_dict(checkpoint_E['model_state_dict'])
    net_G1.load_state_dict(checkpoint_G1['model_state_dict'])
    net_G2.load_state_dict(checkpoint_G2['model_state_dict'])

    net_E.cuda()
    net_G1.cuda()
    net_G2.cuda()

    net_E.eval()
    net_G1.eval()
    net_G2.eval()

    clean = util.imread_uint(os.path.join(opt.dir_test_img, 'clean.png'), 3)
    noisy = util.imread_uint(os.path.join(opt.dir_test_img, 'noisy.png'), 3)
    h, w, _ = clean.shape
    if h%8 != 0 or w%8 != 0:
        clean = clean[:h//8*8, :w//8*8, :]
        noisy = noisy[:h//8*8, :w//8*8, :]

    clean = util.uint2tensor3(clean).unsqueeze(0).cuda()
    noisy = util.uint2tensor3(noisy).unsqueeze(0).cuda()

    with torch.no_grad():
        if opt.NeCA_type == 'W':
            z = torch.randn_like(clean).cuda()
            gain_factor = net_E(noisy)
            pred_noise_level_map = net_G1(clean, gain_factor)
            sdnu_noise = pred_noise_level_map.mul(z)
            sdnc_noise = net_G2(sdnu_noise) + sdnu_noise
            fake_noisy = clean + sdnc_noise

            util.imsave(util.tensor2uint(fake_noisy), os.path.join(opt.dir_test_save, 'fake_noisy.png'))
            util.imsave(util.tensor2uint(sdnc_noise+0.5), os.path.join(opt.dir_test_save, 'sdnc_noise.png'))
            util.imsave(util.tensor2uint(sdnu_noise+0.5), os.path.join(opt.dir_test_save, 'sdnu_noise.png'))

        elif opt.NeCA_type == 'S':
            z = torch.randn_like(clean).cuda()
            z.mul_(opt.noise_level/255.0)
            sinc_noise = net_G2(z) + z
            fake_noisy = clean + sinc_noise

            util.imsave(util.tensor2uint(fake_noisy), os.path.join(opt.dir_test_save, 'fake_noisy.png'))
            util.imsave(util.tensor2uint(sinc_noise+0.5), os.path.join(opt.dir_test_save, 'sinc_noise.png'))


        else:
            raise NotImplementedError('NeCA_type [%s] is not found' % opt.NeCA_type)
        
if __name__ == '__main__':
    main()