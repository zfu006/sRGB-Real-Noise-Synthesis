import torch
import torch.nn as nn
import models.basicblocks as B

class UNet(nn.Module):
    """UNet-based generator"""
    
    def __init__(self, 
                 in_nc=3, 
                 out_nc=3, 
                 nc=64, 
                 act_mode='BR', 
                 num_stages=4, 
                 downsample_mode='strideconv', 
                 upsample_mode='convtranspose', 
                 bias=True,
                 padding_mode='zeros',
                 final_act='Tanh'):
        super(UNet, self).__init__()

        # Define the number of encoder and decoder layers
        num_encoder_layers = num_stages
        num_decoder_layers = num_stages

        # Define the encoder layers
        self.encoders = nn.ModuleList()
        self.encoders.append(B.sequential(B.conv(in_nc, nc, kernel_size=7, padding=3, bias=bias, mode='CR', padding_mode=padding_mode),
                                          B.conv(nc, nc, bias=bias, mode='C'+act_mode)))
        for i in range(1, num_encoder_layers):
            encoder_nc = nc*2**i 
            if encoder_nc > 512: # The maximum channels in the Unet is 512
                break

            if downsample_mode == 'strideconv':
                self.encoders.append(B.sequential(B.conv(encoder_nc//2, encoder_nc, stride=2, bias=bias, mode='C'+act_mode, padding_mode=padding_mode),
                                                  B.conv(encoder_nc, encoder_nc, bias=bias, mode='C'+act_mode, padding_mode=padding_mode)))
            if downsample_mode == 'maxpool':
                self.encoders.append(B.sequential(B.conv(kernel_size=2, stride=2, mode='M'),
                                                  B.conv(encoder_nc//2, encoder_nc, bias=bias, mode='C'+act_mode, padding_mode=padding_mode),
                                                  B.conv(encoder_nc, encoder_nc, bias=bias, mode='C'+act_mode, padding_mode=padding_mode)))
            if downsample_mode == 'avgpool':
                self.encoders.append(B.sequential(B.conv(kernel_size=2, stride=2, mode='A'),
                                                  B.conv(encoder_nc//2, encoder_nc, bias=bias, mode='C'+act_mode, padding_mode=padding_mode),
                                                  B.conv(encoder_nc, encoder_nc, bias=bias, mode='C'+act_mode, padding_mode=padding_mode)))

        self.up_samplers = nn.ModuleList() # This module saves the unet upsampler
        self.decoders = nn.ModuleList() # This module saves the 
        for i in range(1, num_decoder_layers):
            decoder_nc = encoder_nc//(2**i)
            if upsample_mode == 'upsampling_conv':
                # upconv1: Upsampling + 3×3 conv + BN + ReLU, output_channels ---> input_channels/2
                self.up_samplers.append(B.sequential(B.conv(2*decoder_nc, decoder_nc, bias=True, mode='UC'+act_mode, padding_mode=padding_mode)))
                self.decoders.append(B.sequential(B.conv(2*decoder_nc, decoder_nc, bias=True, mode='C'+act_mode, padding_mode=padding_mode),
                                                  B.conv(decoder_nc, decoder_nc, bias=bias, mode='C'+act_mode, padding_mode=padding_mode)))
            
            if upsample_mode == 'upsampling':
                # upconv2: Upsampling
                self.up_samplers.append(B.sequential(B.conv(mode='U')))
                self.decoders.append(B.sequential(B.conv(2*decoder_nc+decoder_nc, decoder_nc, bias=bias, mode='C'+act_mode, padding_mode=padding_mode),
                                                  B.conv(decoder_nc, decoder_nc, bias=bias, mode='C'+act_mode, padding_mode=padding_mode)))
                                    
            if upsample_mode == 'convtranspose':
                # constranspose: nn.convtranspose2d kernel_size=2 stride=2
                self.up_samplers.append(B.sequential(B.conv(2*decoder_nc, decoder_nc, kernel_size=2, stride=2, padding=0, bias=bias, mode='T')))
                self.decoders.append(B.sequential(B.conv(2*decoder_nc, decoder_nc, bias=bias, mode='C'+act_mode, padding_mode=padding_mode),
                                                  B.conv(decoder_nc, decoder_nc, bias=bias, mode='C'+act_mode, padding_mode=padding_mode)))

            if upsample_mode == 'pixelshuffle':
                # pixelshuffle upsampling
                self.up_samplers.append(B.conv(2*decoder_nc, decoder_nc*2**2, bias=bias, mode='C2'))
                self.decoders.append(B.sequential(B.conv(2*decoder_nc, decoder_nc, bias=bias, mode='C'+act_mode, padding_mode=padding_mode),
                                                  B.conv(decoder_nc, decoder_nc, bias=bias, mode='C'+act_mode, padding_mode=padding_mode)))

        # add decoder head 
        self.decoder_head = B.conv(decoder_nc, out_nc, kernel_size=1, padding=0, bias=bias, mode='C')
        self.final_act = final_act
        if final_act == 'Tanh':
            self.final_act_layer = nn.Tanh()
        elif final_act == 'Sigmoid':
            self.final_act_layer = nn.Sigmoid()
        elif final_act == 'exp':
            self.final_act_layer = nn.Identity()
        elif final_act == 'linear':
            self.final_act_layer = nn.Identity()
        else:
            raise NotImplementedError('final activation {:s} is not implemented'.format(final_act))

    def forward(self, x):
        y = x # store the input tensor
        # Pass the input through the encoder layers and store the output
        encoder_outputs = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_outputs.append(x)
        
        # Pass the output through the decoder layers, using the stored encoder outputs as skip connections
        for i, (decoder, up_sampler) in enumerate(zip(self.decoders, self.up_samplers)):
            x = up_sampler(x)
            x = torch.cat((x, encoder_outputs[-i-2]), dim=1) # -i-2 indicates the last second emcoder outputs 
            x = decoder(x)
        
        x = self.decoder_head(x)
        x = self.final_act_layer(x)
        if self.final_act == 'exp':
            x = torch.exp(x)
        
        return x