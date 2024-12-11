import torch 
import torch.nn as nn
import torch.nn.functional as F

def downsample(in_channels, out_channels, kernel_size=4, apply_batchnorm=True):
    layers = []
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=1, bias=False)
    layers.append(conv)
    if apply_batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=False))  # Changed inplace to False
    return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, kernel_size=3):
        super(ResidualBlock, self).__init__()
        stride = 2 if downsample else 1
        padding = 1
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if downsample or in_channels != out_channels:
            self.downconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
            self.downbn = nn.BatchNorm2d(out_channels)
        else:
            self.downconv = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = F.relu(self.bn1(out), inplace=False)  
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downconv is not None:
            identity = self.downconv(identity)
            identity = self.downbn(identity)

        out = out + identity
        out = F.relu(out, inplace=False)  # Changed inplace to False
        return out
class Patches(nn.Module):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size
    
    def forward(self, images):
        B, C, H, W = images.shape
        patches = F.unfold(images, kernel_size=self.patch_size, stride=self.patch_size, padding=0)
        num_patches = patches.shape[-1]
        patch_dims = patches.shape[1]
        # shape : (B, num_patches, C*patch_size*patch_size)
        patches = patches.permute(0, 2, 1)
        return patches

class PatchEncoder(nn.Module):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = nn.Linear(projection_dim, projection_dim)
        self.position_embedding = nn.Embedding(num_patches, projection_dim)

    def forward(self, patch):
        # patch: [B, num_patches, projection_dim]
        positions = torch.arange(self.num_patches, device=patch.device)
        # projection + positional embedding
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded  # [B, num_patches, projection_dim]


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=False)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, num_patches, embed_dim]
        # PyTorch MHA expects [seq_len, batch, embed_dim], so transpose:
        x_t = x.transpose(0, 1)  # [num_patches, B, embed_dim]

        attn_output, _ = self.att(x_t, x_t, x_t)
        attn_output = self.dropout1(attn_output)
        x_t = x_t + attn_output
        x_norm = self.layernorm1(x_t.transpose(0,1))  # back to [B, num_patches, embed_dim]

        ffn_output = self.ffn(x_norm)
        ffn_output = self.dropout2(ffn_output)
        out = x_norm + ffn_output
        out = self.layernorm2(out)
        return out  # [B, num_patches, embed_dim]

class Generator(nn.Module):
    def __init__(self, input_shape=(256,256,1), patch_size=32, num_patches=64, projection_dim=1024, num_heads=4, ff_dim=256):
        super(Generator, self).__init__()

        self.patches = Patches(patch_size)
        self.patch_encoder = PatchEncoder(num_patches, projection_dim)

        self.transformer1 = TransformerBlock(embed_dim=projection_dim, num_heads=num_heads, ff_dim=ff_dim)
        self.transformer2 = TransformerBlock(embed_dim=projection_dim, num_heads=num_heads, ff_dim=ff_dim)
        self.transformer3 = TransformerBlock(embed_dim=projection_dim, num_heads=num_heads, ff_dim=ff_dim)
        self.transformer4 = TransformerBlock(embed_dim=projection_dim, num_heads=num_heads, ff_dim=ff_dim)

        self.convT1 = nn.ConvTranspose2d(projection_dim, 512, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.res1 = ResidualBlock(512, 512, downsample=False)

        self.convT2 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.res2 = ResidualBlock(256, 256, downsample=False)

        self.convT3 = nn.ConvTranspose2d(256, 64, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.res3 = ResidualBlock(64, 64, downsample=False)

        self.convT4 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        self.res4 = ResidualBlock(32, 32, downsample=False)

        self.convT5 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.res5 = ResidualBlock(32, 32, downsample=False)

        self.final_conv = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.loss = nn.BCELoss()

    def forward(self, x):
        patches = self.patches(x)
        encoded_patches = self.patch_encoder(patches)

        x = self.transformer1(encoded_patches)
        x = self.transformer2(x)
        x = self.transformer3(x)
        x = self.transformer4(x)

        B, num_patches, projection_dim = x.shape
        x = x.view(B, 8, 8, projection_dim)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.convT1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2, inplace=False)  # Changed inplace to False
        x = self.res1(x)

        x = self.convT2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2, inplace=False)  # Changed inplace to False
        x = self.res2(x)

        x = self.convT3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2, inplace=False)  # Changed inplace to False
        x = self.res3(x)

        x = self.convT4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x, 0.2, inplace=False)  # Changed inplace to False
        x = self.res4(x)

        x = self.convT5(x)
        x = self.bn5(x)
        x = F.leaky_relu(x, 0.2, inplace=False)  # Changed inplace to False
        x = self.res5(x)

        x = self.final_conv(x)
        x = torch.tanh(x)
        return x

    def generator_loss(self, fake_prediction, fake_ab, real_ab, lamb=100):
        generator_bce = self.loss(fake_prediction, torch.ones_like(fake_prediction))
        l1_loss = F.l1_loss(fake_ab, real_ab)
        return generator_bce + lamb * l1_loss


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.down1 = downsample(3, 64, 4, False)
        self.down2 = downsample(64, 128, 4, True)
        self.down3 = downsample(128, 256, 4, True)

        self.conv = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=False)  
        self.last = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        self.loss = nn.BCELoss()

    def forward(self, inp, tar):
        x = torch.cat([inp, tar], dim=1)

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        x = F.pad(x, (0, 1, 0, 1))
        x = self.conv(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)

        x = F.pad(x, (0, 1, 0, 1))
        x = self.last(x)
        x = torch.sigmoid(x)
        return x

    def discriminator_loss_real(self, real_output):
        real_loss = self.loss(real_output, torch.ones_like(real_output))
        return real_loss
    def discriminator_loss_fake(self, fake_output):
        fake_loss = self.loss(fake_output, torch.zeros_like(fake_output))
        return fake_loss
