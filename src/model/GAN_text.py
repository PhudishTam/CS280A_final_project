import torch 
import torch.nn as nn
import torch.nn.functional as F


class Generator_Unet(nn.Module):
    def __init__(self, input_channels=1, out_channels= 64, kernel_size=4, stride=2, padding=1, use_dropout=False,out_channels_decoder=512,text_embedding_dim=512):
        super().__init__()
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_dropout = use_dropout
        self.out_channels_decoder = out_channels_decoder
        self.text_embedding_dim = text_embedding_dim
        self.conv_layer = nn.Conv2d(input_channels,out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        
        self.conv_block1 = self.conv_block(out_channels, out_channels * 2, kernel_size, stride, padding)
        self.conv_block2 = self.conv_block(out_channels * 2, out_channels * 4, kernel_size, stride, padding)
        self.conv_block3 = self.conv_block(out_channels * 4, out_channels * 8, kernel_size, stride, padding)
        self.conv_block4 = self.conv_block(out_channels * 8, out_channels * 8, kernel_size, stride, padding)
        self.conv_block5 = self.conv_block(out_channels * 8, out_channels * 8, kernel_size, stride, padding)
        self.conv_block6 = self.conv_block(out_channels * 8, out_channels * 8, kernel_size, stride, padding)
        self.conv_block7 = nn.Conv2d(out_channels * 8, out_channels * 8, kernel_size, stride, padding)
        
        self.up_block1 = self.transp_conv_block(512, out_channels_decoder, kernel_size, stride, padding, use_dropout=True)
        self.up_block2 = self.transp_conv_block(out_channels_decoder* 2, out_channels_decoder , kernel_size, stride, padding, use_dropout=True)
        self.up_block3 = self.transp_conv_block( out_channels_decoder* 2, out_channels_decoder, kernel_size, stride, padding, use_dropout=True)
        self.up_block4 = self.transp_conv_block(out_channels_decoder * 2, out_channels_decoder, kernel_size, stride, padding, use_dropout)
        self.up_block5 = self.transp_conv_block(out_channels_decoder * 2, out_channels_decoder // 2, kernel_size, stride, padding, use_dropout)
        self.up_block6 = self.transp_conv_block(out_channels_decoder, out_channels_decoder // 4, kernel_size, stride, padding, use_dropout)
        self.up_block7 = self.transp_conv_block(out_channels_decoder // 2, out_channels_decoder // 8, kernel_size, stride, padding, use_dropout)

        self.up_block8 = nn.ConvTranspose2d(out_channels_decoder // 4, 2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.loss = nn.BCELoss()
        self.text_projection = nn.Sequential(
            nn.Linear(text_embedding_dim, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, 512),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
        ) 

    
    def conv_block(self,in_channels, out_channels, kernel_size, stride, padding):
        block = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        return block
    
    def transp_conv_block(self,in_channels, out_channels, kernel_size, stride, padding,use_dropout=False):
        if use_dropout:
            block = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(0.5),
            )
        else:
            block = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        return block
        
    
    def forward(self, x, text_embedding):
        # Encoder with stored intermediate features
        
        # shape of x is [B, 1, 256, 256] 
        x = x.contiguous()
        # shape of x1 is [B, 1, 256, 256] --> [B, 64, 128, 128]
        x1 = self.conv_layer(x)
        # shape of x2 is [B, 64, 128, 128] --> [B, 128, 64, 64]
        x2 = self.conv_block1(x1)
        # shape of x3 is [B, 128, 64, 64] --> [B, 256, 32, 32]
        x3 = self.conv_block2(x2)
        # shape of x4 is [B, 256, 32, 32] --> [B, 512, 16, 16]
        x4 = self.conv_block3(x3)
        # shape of x5 is [B, 512, 16, 16] --> [B, 512, 8, 8]
        x5 = self.conv_block4(x4)
        # shape of x6 is [B, 512, 8, 8] --> [B, 512, 4, 4]
        x6 = self.conv_block5(x5)
        # shape of x7 is [B, 512, 4, 4] --> [B, 512, 2, 2]
        x7 = self.conv_block6(x6)
        # shape of x8 is [B, 512, 2, 2] --> [B, 512, 1, 1]
        x8 = self.conv_block7(F.leaky_relu(x7, 0.2, inplace=False))
        #print(f"Shape before decoder: {x8.shape}")
        
        # combine with text 
        text_embedding = self.text_projection(text_embedding)
        text_embedding = text_embedding.view(-1, 512, 1, 1)
        x8 = torch.cat([x8, text_embedding], dim=1)
        x8 = self.fusion(x8)
        
        # Decoder
        up1 = self.up_block1(x8)
        up1 = torch.cat([x7, up1], dim=1)
        #print(f"Shape after up1: {up1.shape}")
        up2 = self.up_block2(up1)
        up2 = torch.cat([x6, up2], dim=1)
        up3 = self.up_block3(up2)
        up3 = torch.cat([x5, up3], dim=1)
        up4 = self.up_block4(up3)
        up4 = torch.cat([x4, up4], dim=1)
        up5 = self.up_block5(up4)
        up5 = torch.cat([x3, up5], dim=1)
        up6 = self.up_block6(up5)
        up6 = torch.cat([x2, up6], dim=1)
        up7 = self.up_block7(up6)
        up7 = torch.cat([x1, up7], dim=1)
        up8 = self.up_block8(up7)
        output_image = torch.tanh(up8)
        return output_image
    
    def generator_loss(self, fake_prediction, fake_ab, real_ab):
        generator_bce = self.loss(fake_prediction, torch.ones_like(fake_prediction))
        l1_loss = F.l1_loss(fake_ab, real_ab)
        return generator_bce, l1_loss


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels * 2, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv2_bn = nn.BatchNorm2d(out_channels * 2)
        self.conv3 = nn.Conv2d(out_channels * 2, out_channels * 4, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv3_bn = nn.BatchNorm2d(out_channels * 4)
        self.conv4 = nn.Conv2d(out_channels * 4, out_channels * 8, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.conv4_bn = nn.BatchNorm2d(out_channels * 8)
        self.conv5 = nn.Conv2d(out_channels * 8, 1, kernel_size=kernel_size, stride=1, padding=padding)
        self.loss = nn.BCELoss()

    def forward(self, gray_image,ab_image):
        gray_image = gray_image.contiguous()
        ab_image = ab_image.contiguous()
        x = torch.cat([gray_image, ab_image], dim=1)
        x = F.leaky_relu(self.conv1(x), 0.2, inplace=False)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2, inplace=False)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2, inplace=False)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2, inplace=False)
        x = torch.sigmoid(self.conv5(x))
        return x
        
    def discriminator_loss_real(self, real_output):
        real_loss = self.loss(real_output, torch.ones_like(real_output))
        return real_loss
    def discriminator_loss_fake(self, fake_output):
        fake_loss = self.loss(fake_output, torch.zeros_like(fake_output))
        return fake_loss