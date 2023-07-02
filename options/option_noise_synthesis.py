import argparse

class BaseOptions():
    def initialize(self):
        """Initialize options"""
        parser = argparse.ArgumentParser()

        # basic settings
        parser.add_argument('--dir_save', type=str, default='./saves/noise_syn/', help='training logs are saved here')
        parser.add_argument('--generate_data_lists', action='store_true', help='generate data list for training and testing')
        parser.add_argument('--train_img_dir', type=str, default='./Datasets/SIDD_Medium_Crop/', help='image directory for training data')
        parser.add_argument('--data_prep_dir', type=str, default='./data_preparation/')
        parser.add_argument('--img_channels', type=int, default=3, help='load image channels, 1 for grayscale img and 3 for colorscale img')
        parser.add_argument('--not_train', action='store_true', help='set dataset mode, if not use this flag, the dataset will be the training mode')
        parser.add_argument('--not_aug', action='store_true', help='whether use data augmentation, if not set this flag, the dataset will use augmentation for preproccessing')
        parser.add_argument('--patch_size', type=int, default=96, help='patch size for training')
        parser.add_argument('--cam_name', type=str, default='IP', help='camera name for training data, the dataset will load the images from the specified camera')
        parser.add_argument('--pd_stride', type=int, default=3, help='stride for pixel-shuffle downsampling')
        parser.add_argument('--batch_size', type=int, default=8, help='set batch size for training')
        parser.add_argument('--not_train_shuffle', action='store_true', help='set shuffle scheme for training, if set the flag, the dataset will not be shuffled')
        parser.add_argument('--num_workers', type=int, default=4, help='set number of workers to load data')
        parser.add_argument('--verbose', action='store_true', help='whether print the network information')
        parser.add_argument('--use_pretrained', action='store_true', help='whether use pretrained model')
        parser.add_argument('--use_tb', action='store_true', help='whether use tensorboardX to record training process')
        parser.add_argument('--epochs', type=int, default=5000, help='set the number of epochs for training')
        parser.add_argument('--checkpoint_E_name', type=str, default='E.pth', help='checkpoint name for the GENet')
        parser.add_argument('--checkpoint_G1_name', type=str, default='G1.pth', help='checkpoint name for the NPNet')
        parser.add_argument('--checkpoint_G2_name', type=str, default='G2.pth', help='checkpoint name for the NCNet')
        parser.add_argument('--checkpoint_D1_name', type=str, default='D1.pth', help='checkpoint name for the D1')
        parser.add_argument('--checkpoint_D2_name', type=str, default='D2.pth', help='checkpoint name for the D2')
        parser.add_argument('--checkpoint_G3_name', type=str, default='G3.pth', help='checkpoint name for the denoiser')

        # network and training settings
        parser.add_argument('--E_in_nc', type=int, default=3, help='input channels of the GENet')
        parser.add_argument('--E_out_nc', type=int, default=1, help='output channels of the GENet')
        parser.add_argument('--E_nc', type=int, default=96, help='basic channels of the GENet')
        parser.add_argument('--E_nb', type=int, default=5, help='number of conv layers in the GENet')
        parser.add_argument('--E_act_mode', type=str, default='R', help='GENet layer configuration')
        parser.add_argument('--G1_in_nc', type=int, default=3, help='input channels of the NPNet')
        parser.add_argument('--G1_out_nc', type=int, default=3, help='output channels of the NPNet')
        parser.add_argument('--G1_nc', type=int, default=96, help='basic channels of the NPNet')
        parser.add_argument('--G2_in_nc', type=int, default=3, help='input channels of the NCNet')
        parser.add_argument('--G2_out_nc', type=int, default=3, help='output channels of the NCNet')
        parser.add_argument('--G2_nc', type=int, default=64, help='basic channels of the NCNet')
        parser.add_argument('--G2_act_mode', type=str, default='R', help='NCNet layer configuration')
        parser.add_argument('--G2_downsample_mode', type=str, default='strideconv', help='downsample mode for the NCNet')
        parser.add_argument('--G2_upsample_mode', type=str, default='upsampling', help='upsample mode for the NCNet')
        parser.add_argument('--G2_padding_mode', type=str, default='zeros', help='padding mode for the conv of NCNet')
        parser.add_argument('--G2_final_act', type=str, default='Tanh', choices=['Tanh', 'Sigmoid', 'exp', 'linear'], help='final activation layer for the NCNet')
        parser.add_argument('--D1_in_nc', type=int, default=3, help='input channels of the D1')
        parser.add_argument('--D1_nc', type=int, default=64, help='basic channels of the D1')
        parser.add_argument('--D1_act_mode', type=str, default='L', help='D1 layer configuration')
        parser.add_argument('--D2_in_nc', type=int, default=3, help='input channels of the D2')
        parser.add_argument('--D2_nc', type=int, default=64, help='basic channels of the D2')
        parser.add_argument('--D2_act_mode', type=str, default='L', help='D2 layer configuration')
        parser.add_argument('--G3_in_nc', type=int, default=3, help='input channels of the denoiser')
        parser.add_argument('--G3_out_nc', type=int, default=3, help='output channels of the denoiser')
        parser.add_argument('--G3_nc', type=int, default=64, help='basic channels of the denoiser')
        parser.add_argument('--G3_act_mode', type=str, default='BR', help='denoiser layer configuration')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0], help='set gpu for training, e.g. 0 or 1 for single gpu training and 0 1 for multiple gpus training')
        parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
        parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam')
        parser.add_argument('--lr_E', type=float, default=1e-4, help='learning rate for the GENet')
        parser.add_argument('--lr_G1', type=float, default=1e-4, help='learning rate for the NPNet')
        parser.add_argument('--lr_G2', type=float, default=1e-4, help='learning rate for the NCNet')
        parser.add_argument('--lr_D1', type=float, default=1e-4, help='learning rate for the D1')
        parser.add_argument('--lr_D2', type=float, default=1e-4, help='learning rate for the D2')
        parser.add_argument('--lr_G3', type=float, default=1e-3, help='learning rate for the denoiser')
        parser.add_argument('--lr_decay_1', nargs='+', type=int, default=[1000000], help='learning rate decay for noise synthesis networks at certain epoch')
        parser.add_argument('--lr_decay_2', nargs='+', type=int, default=[100], help='learning rate decay for denoiser at certain epoch')
        parser.add_argument('--joint_start_epoch', type=int, default=50, help='start joint training after epoch')
        parser.add_argument('--seperate_train', action='store_true', help='seperate gradient for training the generator and denoiser')
        parser.add_argument('--gan_mode', type=str, default='wgangp', choices=['wgangp', 'lsgan', 'vanilla'], help='select GAN mode.')
        parser.add_argument('--lambda_reg', type=float, default=1, help='weight for the reg_loss')
        parser.add_argument('--lambda_std1', type=float, default=50, help='weight for the std1_loss')
        parser.add_argument('--lambda_std2', type=float, default=10, help='weight for the std2_loss')
        parser.add_argument('--lambda_adv1', type=float, default=1e-1, help='weight for the adv1_loss')
        parser.add_argument('--lambda_adv2', type=float, default=1e-1, help='weight for the adv2_loss')
        parser.add_argument('--lambda_gp', type=float, default=10, help='weight for the wgangp')
        parser.add_argument('--lambda_rec', type=float, default=1, help='weight for the rec_loss')
        parser.add_argument('--val_step', type=int, default=100000000, help='validation step during training')
        parser.add_argument('--test_epoch', type=int, default=1, help='test epoch during training')
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print options in the log"""
        message = ''
        message += '--------------- Options ---------------\n'

        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: {}]'.format(str(default))
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '--------------- End ---------------\n'
        print(message)

        return message
        
    
    def parse(self):
        opt = self.initialize() 
        opt_message = self.print_options(opt) 
        
        return opt, opt_message

if __name__ == '__main__':
    BaseOptions().parse()