import torch
import torch.nn as nn
import torch.nn.functional as F
 
class Generator_Unet(nn.Module):
    def __init__(self, input_channel, out_channel, kernel_size, stride, padding, out_channel_decoder):
        super(Generator_Unet, self).__init__()
        self.input_channel = input_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.out_channel_decoder = out_channel_decoder
        self.convolution_layer = nn.Conv2d(input_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.convolution_blocks = nn.ModuleList()
        channels = [
            (out_channel, out_channel * 2),
            (out_channel * 2, out_channel * 4),
            (out_channel * 4, out_channel * 8),
            (out_channel * 8, out_channel * 8),
            (out_channel * 8, out_channel * 8),
            (out_channel * 8, out_channel * 8),
            (out_channel * 8, out_channel * 8)
    	]
        for in_channels, out_channels in channels:
            self.convolution_blocks.append(self.convolution_block(in_channels, out_channels, kernel_size, stride, padding))
        self.final_convolution_block = nn.Conv2d(out_channel * 8, out_channel * 8, kernel_size, stride, padding)
    	
        self.transpose_convolution_blocks = nn.ModuleList()
        channels = [
            (out_channel_decoder, out_channel_decoder),
            (out_channel_decoder * 2, out_channel_decoder),
            (out_channel_decoder * 2, out_channel_decoder),
            (out_channel_decoder * 2, out_channel_decoder),
            (out_channel_decoder * 2, out_channel_decoder // 2),
            (out_channel_decoder, out_channel_decoder // 4),
            (out_channel_decoder // 2, out_channel_decoder // 8)]
        dropouts = [True, True, True, False, False, False, False]

        for ((in_channels, out_channels), dropout) in zip(channels, dropouts):
            self.transpose_convolution_blocks.append(self.transpose_convolution_block(in_channels, out_channels, kernel_size, stride, padding, dropout))
    	
        self.final_transpose_convolution = nn.ConvTranspose2d(out_channel_decoder // 4, 2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.loss = nn.BCELoss()
    
    def convolution_block(self, input_channel, output_channel, kernel_size, stride, padding):
        return nn.Sequential(nn.LeakyReLU(0.2, inplace=True),nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                             nn.BatchNorm2d(output_channel))

    def transpose_convolution_block(self, input_channel, output_channel, kernel_size, stride, padding, dropout):
        if dropout:
            return nn.Sequential(nn.ReLU(inplace=True),
                                 nn.ConvTranspose2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                                 nn.BatchNorm2d(output_channel),
                                 nn.Dropout(0.5))
        else:
            return nn.Sequential(nn.ReLU(inplace=True), nn.ConvTranspose2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),nn.BatchNorm2d(output_channel))

    def forward(self, x):
        # Encoder with stored intermediate features
        x = x.contiguous()
        x1 = self.convolution_layer(x)
        x2 = self.convolution_blocks[0](x1)
        x3 = self.convolution_blocks[1](x2)
        x4 = self.convolution_blocks[2](x3)
        x5 = self.convolution_blocks[3](x4)
        x6 = self.convolution_blocks[4](x5)
        x7 = self.convolution_blocks[5](x6)
        x8 = self.convolution_blocks[6](x7)
        
        # Decoder
        up1 = self.transpose_convolution_blocks[0](x8)
        up1 = torch.cat([x7, up1], dim=1)
        up2 = self.transpose_convolution_blocks[1](up1)
        up2 = torch.cat([x6, up2], dim=1)
        up3 = self.transpose_convolution_blocks[2](up2)
        up3 = torch.cat([x5, up3], dim=1)
        up4 = self.transpose_convolution_blocks[3](up3)
        up4 = torch.cat([x4, up4], dim=1)
        up5 = self.transpose_convolution_blocks[4](up4)
        up5 = torch.cat([x3, up5], dim=1)
        up6 = self.transpose_convolution_blocks[5](up5)
        up6 = torch.cat([x2, up6], dim=1)
        up7 = self.transpose_convolution_blocks[6](up6)
        up7 = torch.cat([x1, up7], dim=1)
        up8 = self.final_transpose_convolution(up7)
        output_image = torch.tanh(up8)
        return output_image


class Discriminator(nn.Module):
    def __init__(self, input_channel, out_channel, kernel_size, stride, padding):
        super(Discriminator, self).__init__()
        self.input_channel = input_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.convolution_blocks = nn.ModuleList()
        channels = [
            (input_channel, out_channel),
            (out_channel, out_channel * 2),
            (out_channel * 2, out_channel * 4),
            (out_channel * 4, out_channel * 8)
        ]
        for in_channels, out_channels in channels:
            self.convolution_blocks.append(self.convolution_block(in_channels, out_channels, kernel_size, stride, padding))
        self.final_convolution_block = nn.Conv2d(out_channel * 8, 1, kernel_size, stride, padding)
        self.loss = nn.BCELoss()
    
    
    def convolution_block(self, input_channel, output_channel, kernel_size, stride, padding):
        return nn.Sequential(nn.LeakyReLU(0.2, inplace=True), 
                             nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False), 
                             nn.BatchNorm2d(output_channel))
    
    def forward(self, x):
        x = x.contiguous()
        x1 = self.convolution_blocks[0](x)
        x2 = self.convolution_blocks[1](x1)
        x3 = self.convolution_blocks[2](x2)
        x4 = self.convolution_blocks[3](x3)
        x5 = self.final_convolution_block(x4)
        return torch.sigmoid(x5)
    
    def discriminator_loss_real(self, real_output):
        real_loss = self.loss(real_output, torch.ones_like(real_output))
        return real_loss
    def discriminator_loss_fake(self, fake_output):
        fake_loss = self.loss(fake_output, torch.zeros_like(fake_output))
        return fake_loss    
