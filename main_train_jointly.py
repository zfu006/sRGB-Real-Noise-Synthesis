import torch
import torch.nn as nn
import logging
import os
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

from options.option_noise_synthesis import BaseOptions
from utils.utils_logger import logger_info
from data.dataset_noise_synthesis import dataset_noise_synthesis
from models.network_unet import UNet
from models.network_plain import Noise_estimator, Noise_level_predictor, DnCNN 
from models.network_discriminator import Discriminator_96, Discriminator_32 
import utils.utils_network as net_util
import utils.utils_image as util


if __name__ == '__main__':
    """
    # -----------------------------------------------
    # step 1 -- initiate logs and args
    # -----------------------------------------------
    """
    # get training options
    opt, opt_message = BaseOptions().parse() 

    # initiate data folder
    cam_name = opt.cam_name
    dir_save_logs = os.path.join(opt.dir_save, cam_name, 'logs/')
    dir_save_imgs = os.path.join(opt.dir_save, cam_name, 'imgs/')
    dir_save_models = os.path.join(opt.dir_save, cam_name, 'models/')
    dir_save_tb = os.path.join(opt.dir_save, cam_name, 'tb/')

    if not os.path.exists(dir_save_logs):
        os.makedirs(dir_save_logs)
    if not os.path.exists(dir_save_imgs):
        os.makedirs(dir_save_imgs)
    if not os.path.exists(dir_save_models):
        os.makedirs(dir_save_models)
    if not os.path.exists(dir_save_tb):
        os.makedirs(dir_save_tb)

    # initiate logs
    logger_name = 'train'
    logger_info(logger_name, log_path=dir_save_logs+logger_name+'.log')
    logger = logging.getLogger(logger_name)
    logger.info(opt_message)
    
    """
    # -----------------------------------------------
    # step 2 -- prepare dataloader
    # -----------------------------------------------
    """
    if opt.generate_data_lists:
        util.train_test_split(opt.train_img_dir, opt.data_prep_dir, cam_name=cam_name, ratio=0.8)
    
    train_img_lists, val_img_lists = util.get_img_list(opt.train_img_dir, cam_name=cam_name, mode='train', ratio=1)
    test_img_lists = util.get_img_list(opt.train_img_dir, cam_name=cam_name, mode='test')
    data_message = 'train: {:d}, val: {:d}, test: {:d}'.format(len(train_img_lists), len(val_img_lists), len(test_img_lists))
    print(data_message)
    logger.info(data_message)

    train_data = dataset_noise_synthesis(img_lists=train_img_lists,
                                         img_channels=opt.img_channels,
                                         not_train=opt.not_train,
                                         not_aug=opt.not_aug,
                                         patch_size=opt.patch_size)
    val_data = dataset_noise_synthesis(img_lists=val_img_lists,
                                       img_channels=opt.img_channels,
                                       not_train=True)
    test_data = dataset_noise_synthesis(img_lists=test_img_lists,
                                        img_channels=opt.img_channels,
                                        not_train=True)
    
    train_loader = DataLoader(dataset=train_data,
                              batch_size=opt.batch_size,
                              shuffle=not opt.not_train_shuffle,
                              num_workers=opt.num_workers,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(dataset=val_data,
                            batch_size=1,
                            shuffle=False,
                            num_workers=opt.num_workers,
                            pin_memory=True,
                            drop_last=False)
    test_loader = DataLoader(dataset=test_data,
                             batch_size=1,
                             shuffle=False,
                             num_workers=opt.num_workers,
                             pin_memory=True,
                             drop_last=False)                            
    
    """
    # ------------------------------------------------
    # step 3 -- prepare networks, optims and loss
    # ------------------------------------------------ 
    """

    """prepare networks
    net_E -- Gain estimation network (GENet)
    net_G1 -- Noise-level prediction network (NPNet)
    net_G2 -- Neighboring correlation network (NCNet)
    net_D1 -- Discriminator for neighboring uncorrelated noise
    net_D2 -- Discriminator for neighboring correlated noise
    net_G3 -- Denoiser for real noise, used for joint training
    """
    net_E = Noise_estimator(in_nc=opt.E_in_nc, 
                            out_nc=opt.E_out_nc, 
                            nc=opt.E_nc, 
                            nb=opt.E_nb, 
                            act_mode=opt.E_act_mode)
    
    net_G1 = Noise_level_predictor(in_nc=opt.G1_in_nc, out_nc=opt.G1_out_nc, nc=opt.G1_nc)

    net_G2 = UNet(in_nc=opt.G2_in_nc,
                 out_nc=opt.G2_out_nc,
                 nc=opt.G2_nc,
                 act_mode=opt.G2_act_mode,
                 downsample_mode=opt.G2_downsample_mode,
                 upsample_mode=opt.G2_upsample_mode,
                 padding_mode=opt.G2_padding_mode,
                 final_act=opt.G2_final_act)
    
    net_D1 = Discriminator_96(in_nc=opt.D1_in_nc, nc=opt.D1_nc, act_mode=opt.D1_act_mode)
    
    net_D2 = Discriminator_32(in_nc=opt.D2_in_nc, nc=opt.D2_nc, act_mode=opt.D2_act_mode)

    net_G3 = DnCNN(in_nc=opt.G3_in_nc, 
                   out_nc=opt.G3_out_nc,
                   nc=opt.G3_nc,
                   act_mode=opt.G3_act_mode)

    # print networks
    net_message = '------------ Network params -----------\n'
    net_message += net_util.print_networks(network=net_E, network_name='GENet', verbose=opt.verbose)
    net_message += net_util.print_networks(network=net_G1, network_name='NPNet', verbose=opt.verbose)
    net_message += net_util.print_networks(network=net_G2, network_name='NCNet', verbose=opt.verbose)
    net_message += net_util.print_networks(network=net_D1, network_name='D1', verbose=opt.verbose)
    net_message += net_util.print_networks(network=net_D2, network_name='D2', verbose=opt.verbose)
    net_message += net_util.print_networks(network=net_G3, network_name='Denoiser', verbose=opt.verbose)
    net_message += '-----------------------------------------------\n'
    logger.info(net_message)

    # send networks to GPU
    net_E, gpu_message, device = net_util.model_to_device(net_E, gpu_ids=opt.gpu_ids)
    net_G1, _, _ = net_util.model_to_device(net_G1, gpu_ids=opt.gpu_ids)
    net_G2, _, _ = net_util.model_to_device(net_G2, gpu_ids=opt.gpu_ids)
    net_D1, _, _ = net_util.model_to_device(net_D1, gpu_ids=opt.gpu_ids)
    net_D2, _, _ = net_util.model_to_device(net_D2, gpu_ids=opt.gpu_ids)
    net_G3, _, _ = net_util.model_to_device(net_G3, gpu_ids=opt.gpu_ids)
    logger.info(gpu_message)

    # load pretrained networks 
    if opt.use_pretrained:
        checkpoint_E = torch.load(os.path.join(dir_save_models, opt.checkpoint_E_name), map_location=device)
        checkpoint_G1 = torch.load(os.path.join(dir_save_models, opt.checkpoint_G1_name), map_location=device)
        checkpoint_G2 = torch.load(os.path.join(dir_save_models, opt.checkpoint_G2_name), map_location=device)
        checkpoint_D1 = torch.load(os.path.join(dir_save_models, opt.checkpoint_D1_name), map_location=device)
        checkpoint_D2 = torch.load(os.path.join(dir_save_models, opt.checkpoint_D2_name), map_location=device)
        # checkpoint_G3 = torch.load(os.path.join(dir_save_models, opt.checkpoint_G3_name), map_location=device)
        net_E.load_state_dict(checkpoint_E['model_state_dict'])
        net_G1.load_state_dict(checkpoint_G1['model_state_dict'])
        net_G2.load_state_dict(checkpoint_G2['model_state_dict'])
        net_D1.load_state_dict(checkpoint_D1['model_state_dict'])
        net_D2.load_state_dict(checkpoint_D2['model_state_dict'])
        # net_G3.load_state_dict(checkpoint_G3['model_state_dict'])

    else:
        net_util.init_weights(net_E, init_type=opt.init_type)
        net_util.init_weights(net_G1, init_type=opt.init_type)
        net_util.init_weights(net_G2, init_type=opt.init_type)
        net_util.init_weights(net_D1, init_type=opt.init_type)
        net_util.init_weights(net_D2, init_type=opt.init_type)
        net_util.init_weights(net_G3, init_type=opt.init_type)
    

    # set optimizers
    optim_E = optim.Adam(net_E.parameters(), lr=opt.lr_E, betas=(opt.beta1, opt.beta2))
    optim_G1 = optim.Adam(net_G1.parameters(), lr=opt.lr_G1, betas=(opt.beta1, opt.beta2))
    optim_G2 = optim.Adam(net_G2.parameters(), lr=opt.lr_G2, betas=(opt.beta1, opt.beta2))
    optim_D1 = optim.Adam(net_D1.parameters(), lr=opt.lr_D1, betas=(opt.beta1, opt.beta2))
    optim_D2 = optim.Adam(net_D2.parameters(), lr=opt.lr_D2, betas=(opt.beta1, opt.beta2))
    optim_G3 = optim.Adam(net_G3.parameters(), lr=opt.lr_G3, betas=(opt.beta1, opt.beta2))

    # set learning rate schedulers
    lr_scheduler_E = optim.lr_scheduler.MultiStepLR(optimizer=optim_E, milestones=opt.lr_decay_1, gamma=0.1)
    lr_scheduler_G1 = optim.lr_scheduler.MultiStepLR(optimizer=optim_G1, milestones=opt.lr_decay_1, gamma=0.1)
    lr_scheduler_G2 = optim.lr_scheduler.MultiStepLR(optimizer=optim_G2, milestones=opt.lr_decay_1, gamma=0.1)
    lr_scheduler_D1 = optim.lr_scheduler.MultiStepLR(optimizer=optim_D1, milestones=opt.lr_decay_1, gamma=0.1)
    lr_scheduler_D2 = optim.lr_scheduler.MultiStepLR(optimizer=optim_D2, milestones=opt.lr_decay_1, gamma=0.1)
    lr_scheduler_G3 = optim.lr_scheduler.MultiStepLR(optimizer=optim_G3, milestones=opt.lr_decay_2, gamma=0.1)

    # set loss functions
    Cri_rec = nn.MSELoss()
    Cri_gan = net_util.GANLoss(gan_mode=opt.gan_mode).to(device)

    # set tensorboard writer
    if opt.use_tb:
        tb = SummaryWriter(log_dir=dir_save_tb)
    
    """
    # ------------------------------------------------
    # step 4 -- training
    # ------------------------------------------------
    """
    current_step = 0
    start_epoch = 1 if not opt.use_pretrained else checkpoint_E['epoch']
    iter_step = 5 if opt.gan_mode == 'wgangp' else 1
    log_step = iter_step
    for epoch in range(start_epoch, opt.epochs+1):
        for i, data in enumerate(train_loader):
            
            current_step += 1
            clean = data['clean'].to(device)
            noisy = data['noisy'].to(device)
            noise = data['noise'].to(device)
            training_message = OrderedDict()
            
            # 1) feed data
            z = torch.randn_like(clean).to(device)
            gain_factor = net_E(noisy) # estimate gain factor from noisy image
            pred_noise_level_map = net_G1(clean, gain_factor) 
            sdnu_noise = pred_noise_level_map.mul(z) # synthesize signal-dependent neighboring uncorrelated noise
            sdnc_noise = net_G2(sdnu_noise) + sdnu_noise # synthesize signal-dependent neighboring correlated noise


            global_nl = torch.std(noise, dim=(1, 2, 3), keepdim=True) # global estimated noise level
            mean_filter = nn.AvgPool2d((7, 7), stride=1, padding=3)
            local_nl = torch.sqrt(mean_filter(noise**2) - (mean_filter(noise))**2) # local estimated noise level, var=E(x^2)-E(x)^2
            pixel_shuffle = util.pixel_shuffle(stride=opt.pd_stride)
            sub_noise = pixel_shuffle._crop(noise)
            sub_sdnu_noise = pixel_shuffle._crop(sdnu_noise)

            if opt.joint_start_epoch <= epoch:
                if opt.seperate_train:
                    fake_noisy = sdnc_noise.detach() + clean
                else:
                    fake_noisy = sdnc_noise + clean
                rec_clean = net_G3(fake_noisy)

            # 2) optimize generator
            optim_E.zero_grad()
            optim_G1.zero_grad()
            optim_G2.zero_grad()
            optim_G3.zero_grad()

            std1_loss = Cri_rec(gain_factor, global_nl)
            std2_loss = Cri_rec(pred_noise_level_map, local_nl)
            reg_loss = Cri_rec(pred_noise_level_map, gain_factor.expand_as(pred_noise_level_map))
            G_losses = opt.lambda_reg * reg_loss + opt.lambda_std1 * std1_loss + opt.lambda_std2 * std2_loss
            if opt.joint_start_epoch <= epoch:
                rec_loss = Cri_rec(rec_clean, clean)
                G_losses += opt.lambda_rec * rec_loss

            if current_step % iter_step == 0:

                fake1 = net_D1(sdnc_noise)
                fake2  =net_D2(sub_sdnu_noise)
                adv1_loss = Cri_gan(fake1, True)
                adv2_loss = Cri_gan(fake2, True)
                G_losses += opt.lambda_adv1 * adv1_loss + opt.lambda_adv2 * adv2_loss
            
            G_losses.backward()
            optim_E.step()
            optim_G1.step()
            optim_G2.step()
            optim_G3.step() 

            # 3) optimize discriminator
            optim_D1.zero_grad()
            optim_D2.zero_grad()

            real1 = net_D1(noise)
            fake1 = net_D1(sdnc_noise.detach())
            real2 = net_D2(sub_noise)
            fake2 = net_D2(sub_sdnu_noise.detach())
            D_r_loss = Cri_gan(real1, True) + Cri_gan(real2, True)
            D_f_loss = Cri_gan(fake1, False) + Cri_gan(fake2, False)
            if opt.gan_mode == 'wgangp':
                gp = net_util.gradient_penalty(net_D1, noise, sdnc_noise.detach(), device) + net_util.gradient_penalty(net_D2, sub_noise, sub_sdnu_noise.detach(), device)
                D_losses = D_r_loss + D_f_loss+ opt.lambda_gp * gp
            else:
                D_losses = 0.5*(D_r_loss + D_f_loss)
            D_losses.backward()
            optim_D1.step()
            optim_D2.step()

            # 4) logs
            if current_step % log_step == 0:
                training_message['reg_loss'] = reg_loss.item()
                training_message['std1_loss'] = std1_loss.item()
                training_message['std2_loss'] = std2_loss.item()
                training_message['adv1_loss'] = adv1_loss.item()
                training_message['adv2_loss'] = adv2_loss.item()
                training_message['D_r_loss'] = D_r_loss.item()
                training_message['D_f_loss'] = D_f_loss.item()
                loss_message = '<epoch:{:d}, step: {:08d}> '.format(epoch, current_step)
                
                for k, v in training_message.items():  # merge log information into message
                    loss_message += '{:s}: {:.3e} '.format(k, v)
                print(loss_message)
                logger.info(loss_message)

                if opt.use_tb:
                    tb.add_scalar('reg_loss', training_message['reg_loss'], current_step)
                    tb.add_scalar('std1_loss', training_message['std1_loss'], current_step)
                    tb.add_scalar('std2_loss', training_message['std2_loss'], current_step)
                    tb.add_scalar('adv1_loss', training_message['adv1_loss'], current_step)
                    tb.add_scalar('adv2_loss', training_message['adv2_loss'], current_step)
                    tb.add_scalar('D_r_loss', training_message['D_r_loss'], current_step)
                    tb.add_scalar('D_f_loss', training_message['D_f_loss'], current_step)
            
            # if current_step % opt.val_step == 0:
            #     # validation
            #     net_E.eval()
            #     net_G1.eval()
            #     net_G2.eval()
            #     net_G3.eval()
            #     val_message = OrderedDict()
            #     val_kl = 0
            #     val_psnr = 0
            #     val_num = 0

            #     for data in val_loader: 
            #         val_num += 1
            #         clean = data['clean'].to(device)
            #         noisy = data['noisy'].to(device)
            #         noise = data['noise'].to(device)
            #         with torch.no_grad():
            #             z = torch.randn_like(clean).to(device)
            #             gain_factor = net_E(noisy)
            #             pred_noise_level_map = net_G1(clean, gain_factor)
            #             sdnu_noise = pred_noise_level_map.mul(z)
            #             sdnc_noise = sdnu_noise + net_G2(sdnu_noise)
            #             fake_noisy = clean + sdnc_noise
            #             quantized_noise = util.noise_quantization(fake_noisy, clean)
            #             if opt.joint_start_epoch <= epoch:
            #                 rec_clean = net_G3(noisy)

            #             val_kl += util.cal_kld(util.tensor2data(noise), util.tensor2data(quantized_noise))
            #             if opt.joint_start_epoch <= epoch:
            #                 val_psnr += util.calculate_psnr(util.tensor2uint(rec_clean), util.tensor2uint(clean))

            #     val_kl /= val_num
            #     val_psnr /= val_num
            #     val_message['val_kl'] = val_kl
            #     val_message['val_psnr'] = val_psnr
            #     message = '<epoch:{:d}, step: {:08d}> '.format(epoch, current_step)

            #     for k, v in val_message.items():
            #         message += '{:s}: {:.3e} '.format(k, v)
            #     print(message)
            #     logger.info(message)
            #     if opt.use_tb:
            #         tb.add_scalar('val_kl', val_kl, current_step)
            #         tb.add_scalar('val_psnr', val_psnr, current_step)

            #     net_E.train()
            #     net_G1.train()
            #     net_G2.train()
            #     net_G3.train()
        
        # test
        if epoch % opt.test_epoch == 0:
            net_E.eval()
            net_G1.eval()
            net_G2.eval()
            net_G3.eval()
            test_message = OrderedDict()
            test_kl = 0
            test_psnr = 0
            test_num = 0

            for data in test_loader:
                test_num += 1
                clean = data['clean'].to(device)
                noisy = data['noisy'].to(device)
                noise = data['noise'].to(device)
                with torch.no_grad():
                    z = torch.randn_like(clean).to(device)
                    gain_factor = net_E(noisy)
                    pred_noise_level_map = net_G1(clean, gain_factor)
                    sdnu_noise = pred_noise_level_map.mul(z)
                    sdnc_noise = sdnu_noise + net_G2(sdnu_noise)
                    fake_noisy = clean + sdnc_noise
                    quantized_noise = util.noise_quantization(fake_noisy, clean)

                    if opt.use_tb:
                        tb.add_image('clean', clean.squeeze(), epoch)
                        tb.add_image('noisy', noisy.squeeze(), epoch)
                        tb.add_image('noise', noise.squeeze(), epoch)
                        tb.add_image('sdnu_noise', sdnu_noise.squeeze()+0.5, epoch)
                        tb.add_image('sdnc_noise', sdnc_noise.squeeze()+0.5, epoch)
                        tb.add_image('fake_noisy', fake_noisy.squeeze(), epoch)
                    
                    if opt.joint_start_epoch <= epoch:
                        rec_clean = net_G3(noisy)

                    test_kl += util.cal_kld(util.tensor2data(noise), util.tensor2data(quantized_noise))
                    if opt.joint_start_epoch <= epoch:
                        test_psnr += util.calculate_psnr(util.tensor2uint(rec_clean), util.tensor2uint(clean))

                if opt.joint_start_epoch <= epoch and opt.use_tb:
                    tb.add_image('rec_clean', rec_clean.squeeze(), epoch)

            test_kl /= test_num
            test_psnr /= test_num
            test_message['test_kl'] = test_kl
            test_message['test_psnr'] = test_psnr
            message = '<epoch:{:d}> '.format(epoch)

            for k, v in test_message.items():
                message += '{:s}: {:.3e} '.format(k, v)
            print(message)
            logger.info(message)
            if opt.use_tb:
                tb.add_scalar('test_kl', test_kl, current_step)
                tb.add_scalar('test_psnr', test_psnr, current_step)
            
            net_E.train()
            net_G1.train()
            net_G2.train()
            net_G3.train()

            # save models
            checkpoint_E = {'epoch': epoch, 'model_state_dict': net_util.get_bare_model(net_E).state_dict()}
            checkpoint_G1 = {'epoch': epoch, 'model_state_dict': net_util.get_bare_model(net_G1).state_dict()}
            checkpoint_G2 = {'epoch': epoch, 'model_state_dict': net_util.get_bare_model(net_G2).state_dict()}
            checkpoint_D1 = {'epoch': epoch, 'model_state_dict': net_util.get_bare_model(net_D1).state_dict()}
            checkpoint_D2 = {'epoch': epoch, 'model_state_dict': net_util.get_bare_model(net_D2).state_dict()}
            checkpoint_G3 = {'epoch': epoch, 'model_state_dict': net_util.get_bare_model(net_G3).state_dict()}
            torch.save(checkpoint_E, os.path.join(dir_save_models, 'checkpoint_E_{:08d}.pth'.format(epoch)))
            torch.save(checkpoint_G1, os.path.join(dir_save_models, 'checkpoint_G1_{:08d}.pth'.format(epoch)))
            torch.save(checkpoint_G2, os.path.join(dir_save_models, 'checkpoint_G2_{:08d}.pth'.format(epoch)))
            torch.save(checkpoint_D1, os.path.join(dir_save_models, 'checkpoint_D1_{:08d}.pth'.format(epoch)))
            torch.save(checkpoint_D2, os.path.join(dir_save_models, 'checkpoint_D2_{:08d}.pth'.format(epoch)))
            torch.save(checkpoint_G3, os.path.join(dir_save_models, 'checkpoint_G3_{:08d}.pth'.format(epoch)))

        # update learning rate
        lr_scheduler_E.step()
        lr_scheduler_G1.step()
        lr_scheduler_G2.step()
        lr_scheduler_D1.step()
        lr_scheduler_D2.step()  
        lr_scheduler_G3.step()
            
