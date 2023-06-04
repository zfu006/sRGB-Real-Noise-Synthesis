import torch
import torch.nn as nn
import models.basicblocks as B

# --------------------------------------------
# Networks for noise synthesis
# --------------------------------------------

class Noise_estimator(nn.Module):
    """Gain estimation network -- Estimate global noise level from noisy images"""

    def __init__(self,
                 in_nc=3,
                 out_nc=3,
                 nc=96,
                 nb=5,
                 act_mode='R'):
        super(Noise_estimator, self).__init__()
        model = []
        model += [B.conv(in_nc, nc, 7, padding = 3, mode='C'+act_mode[0])]
        model += [B.conv(nc, nc, mode='C'+act_mode[0]) for _ in range(nb-2)]
        linear = [B.conv(nc, out_nc, 1, 1, 0, mode='C')]

        self.model = nn.Sequential(*model)
        self.linear = nn.Sequential(*linear)

    def forward(self, x):
        y = self.model(x)
        y = nn.AdaptiveAvgPool2d((1, 1))(y) 
        y = self.linear(y)

        return y
    
class Noise_level_predictor(nn.Module):
    """Noise level prediction network -- Estimate pixel-wise noise level for a clean image"""

    def __init__(self,
                 in_nc=3,
                 out_nc=3,
                 nc=96,
                 ):
        
        super(Noise_level_predictor, self).__init__()
        self.conv1 = nn.Conv2d(in_nc, nc, 7, 1, 3, padding_mode='reflect')

        self.conv2 = nn.Conv2d(nc, nc, 1, 1, 0)
        self.act2 = nn.LeakyReLU()
        
        self.conv3 = nn.Conv2d(nc, nc, 1, 1, 0)
        self.act3 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(nc, nc, 1, 1, 0)
        self.act4 = nn.LeakyReLU()

        self.conv5 = nn.Conv2d(nc, out_nc, 1, 1, 0)

    def forward(self, x, y):
        x = self.conv1(x)

        x = self.conv2(x)
        mean2 = torch.mean(x, dim=[1, 2, 3], keepdim=True)
        var2 = torch.std(x, dim=[1, 2, 3], keepdim=True)
        x = (x-mean2)/(var2+1e-5)
        x = x.mul(y)
        x = self.act2(x)

        x = self.conv3(x)
        mean3 = torch.mean(x, dim=[1, 2, 3], keepdim=True)
        var3 = torch.std(x, dim=[1, 2, 3], keepdim=True)
        x = (x-mean3)/(var3+1e-5)
        x = x.mul(y)
        x = self.act3(x)
        
        x = self.conv4(x)
        mean4 = torch.mean(x, dim=[1, 2, 3], keepdim=True)
        var4 = torch.std(x, dim=[1, 2, 3], keepdim=True)
        x = (x-mean4)/(var4+1e-5)
        x = x.mul(y)
        x = self.act4(x)

        x = self.conv5(x)
        x = x**2
        
        return x

# --------------------------------------------
# Networks for denoising
# --------------------------------------------
class DnCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=20, act_mode='BR'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(DnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = B.conv(nc, out_nc, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)


    def forward(self, x):
        n = self.model(x)
        return x[:, :3, :, :] - n
