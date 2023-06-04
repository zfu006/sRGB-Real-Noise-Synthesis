import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.nn import init


"""
# --------------------------------------------
# GPU parallel setup
# --------------------------------------------
# """

def model_to_device(network, gpu_ids=[0]):
    device = torch.device('cuda:{}'.format(gpu_ids[0]) if gpu_ids and torch.cuda.is_available() else 'cpu')  
    message = ''
    message += '--------------------------------------------\n'
    
    if torch.cuda.device_count()==0:
        message += 'use cpu.\n'
    elif len(gpu_ids)==1 or torch.cuda.device_count()==1:
        message += 'use single gpu.\n'
    else:
        network = DataParallel(network, device_ids=gpu_ids)
        message += 'use multiple-gpus (DataParallel).\n'

    message += '--------------------------------------------\n'
    network.to(device)
    print(message)

    return network, message, device

"""
# -------------------------------------------
# init networks
# -------------------------------------------
"""
def init_weights(network, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        network (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s\n' % init_type)
    network.apply(init_func)  # apply the initialization function <init_func>

    return network

"""
# -----------------------------------------------
# get bare model, unwrap models from Dataparallel
# -----------------------------------------------
"""
def get_bare_model(network):
    if isinstance(network, (DataParallel, DistributedDataParallel)):
        network = network.module
    return network

"""
# --------------------------------------------
# print network params 
# --------------------------------------------
"""
def print_networks(network, network_name, verbose=False):
    """Print the total number of parameters in the network and (if verbose) network architecture

    Parameters:
    verbose (bool) -- if verbose: print the network architecture
    """
    network = get_bare_model(network=network)
    message = ''
    # calculate the number of network params
    message += '[Network {}] params number: {:.3f} M'.format(network_name, sum(map(lambda x: x.numel(), network.parameters()))/1e6) + '\n'
    
    # print network structure for debug
    if verbose:
        message += 'Net structure:\n{}'.format(str(network)) + '\n'
   
    print(message)

    return message

"""
# --------------------------------------------
# GAN loss
# --------------------------------------------
"""
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)
        
    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - typically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and ground truth labels.

        Parameters:
            prediction (tensor) - - typically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
        
def gradient_penalty(disc, real, fake, device):
    """Calculate the gradient penalty loss"""
    b, c, h, w = real.size()
    epsilon = torch.rand((b, 1, 1, 1)).to(device)
    mixed_images = real * epsilon + fake * (1 - epsilon)
    mixed_images.requires_grad_(True)
    mixed_scores = disc(mixed_images)

    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True)[0]
    
    gradient = gradient.view(gradient.size(0), -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1)**2)

    return gradient_penalty
