import os
import utils.utils_image as util

def main():
    ori_dir = './Datasets/SIDD_Medium_Srgb_Parts/' # put original SIDD directory here
    target_dir = './Datasets/SIDD_Medium_Crop/' # put target cropped SIDD directory here

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    
    util.crop_sidd_dataset(ori_dir, target_dir, p_size=512, p_overlap=128, p_max=800)

if __name__ == '__main__':
    main()