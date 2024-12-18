import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import json
import sys
import torch.optim as optim
sys.path.append("/accounts/grad/phudish_p/CS280A_final_project/src")
from pre_process_data.dataset import Datasetcoloritzation

def transpose_convolution_block(input_channel, output_channel, kernel_size, stride, padding, dropout):
        if dropout:
            return nn.Sequential(nn.ReLU(inplace=False),nn.ConvTranspose2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),nn.BatchNorm2d(output_channel),
                                 nn.Dropout(0.5))
        else:
            return nn.Sequential(nn.ReLU(inplace=False), nn.ConvTranspose2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),nn.BatchNorm2d(output_channel))
        
class VectorQuantization(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).reshape(-1, C)

        distances = torch.cdist(x, self.codebook.weight)
        min_distances = torch.argmin(distances, dim=1)
        z_q = self.codebook.weight[min_distances].view(batch_size, H, W, C).permute(0, 3, 1, 2)

        vq_objective = torch.mean((x.detach() - self.codebook.weight[min_distances]) ** 2)
        commitment_loss = torch.mean((x - self.codebook.weight[min_distances].detach()) ** 2)
        vq_loss = vq_objective + self.commitment_cost * commitment_loss

        return z_q, vq_loss

class VQResNetUNet(nn.Module):
    def __init__(self, num_embeddings, commitment_cost, out_channels_decoder):
        super().__init__()
        self.resent = models.resnet34(pretrained=True)
        self.encoder_0 = nn.Sequential(
            self.resent.conv1,
            self.resent.bn1,
            self.resent.relu)
        self.encoder_1 = nn.Sequential(
            self.resent.maxpool,
            self.resent.layer1)
        self.encoder_2 = self.resent.layer2
        self.encoder_3 = self.resent.layer3
        self.encoder_4 = self.resent.layer4
        # quantizer 
        self.quantizer1 = VectorQuantization(num_embeddings, 64, commitment_cost)
        self.quantizer2 = VectorQuantization(num_embeddings, 128, commitment_cost)
        self.quantizer3 = VectorQuantization(num_embeddings, 256, commitment_cost)
        self.quantizer4 = VectorQuantization(num_embeddings, 512, commitment_cost)   
        # Decoder
        self.up_block1 = transpose_convolution_block(out_channels_decoder, out_channels_decoder // 2, 4, 2, 1, dropout=False)
        self.up_block2 = transpose_convolution_block(out_channels_decoder, out_channels_decoder // 4, 4, 2, 1, dropout=True) 
        self.up_block3 = transpose_convolution_block(out_channels_decoder // 2, out_channels_decoder // 8, 4, 2, 1,dropout=True) 
        self.up_block4 = transpose_convolution_block(out_channels_decoder // 4, out_channels_decoder // 8, 4, 2, 1, dropout=True) 
        self.up_block5 = transpose_convolution_block(out_channels_decoder // 4, 2, 4, 2, 1, dropout=False)
        self.loss = nn.BCELoss()

    def forward(self, x):
        # Encoder
        e_0 = self.encoder_0(x)
        e_1 = self.encoder_1(e_0)
        q1, vq_loss1 = self.quantizer1(e_1)
        e_2 = self.encoder_2(q1)
        q2, vq_loss2 = self.quantizer2(e_2)
        e_3 = self.encoder_3(q2)
        q3, vq_loss3 = self.quantizer3(e_3)
        e_4 = self.encoder_4(q3)
        q4, vq_loss4 = self.quantizer4(e_4)
        # Decoder
        up1 = self.up_block1(q4)
        up1 = torch.cat([e_3, up1], dim=1)
        up2 = self.up_block2(up1)
        up2 = torch.cat([e_2, up2], dim=1)
        up3 = self.up_block3(up2)
        up3 = torch.cat([e_1, up3], dim=1)
        up4 = self.up_block4(up3)
        up4 = torch.cat([e_0, up4], dim=1)
        up4 = F.relu(up4, inplace=True)
        up5 = self.up_block5(up4)
        output_image = torch.tanh(up5)
        return output_image, vq_loss1 + vq_loss2 + vq_loss3 + vq_loss4
       
    def generator_loss(self, fake_prediction, fake_ab, real_ab):
        generator_bce = self.loss(fake_prediction, torch.ones_like(fake_prediction))
        l1_loss = F.l1_loss(fake_ab, real_ab)
        return generator_bce, l1_loss

def pretrain_generator(generator, train_loader, optimizer, criterion, device, epochs):
    generator.train()
    loss_history = []
    for epoch in range(epochs):
        loss_meter = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            gray_images = batch["l_channels"].to(device).permute(0, 3, 1, 2)  
            ab_images = batch["ab_channels"].to(device).permute(0, 3, 1, 2)    
            gray_images_pretrain = gray_images.repeat(1, 3, 1, 1)
            preds, vq_loss = generator(gray_images_pretrain)
            loss = criterion(preds, ab_images) + vq_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter += loss.item()
        avg_loss = loss_meter / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")
    with open("pretraining_vq_GAN.json", "w") as f:
        json.dump(loss_history, f)
    

def normal_init(weight, mean=0.0, std=0.02):
    if isinstance(weight, (nn.ConvTranspose2d, nn.Conv2d)):
        nn.init.normal_(weight.weight, mean=mean, std=std)
        if weight.bias is not None:
            nn.init.constant_(weight.bias, 0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    json_file_path = "/accounts/grad/phudish_p/CS280A_final_project/src/hparams/GAN.json"
    with open(json_file_path, "r") as f:
        hparams = json.load(f)
    train_data_dir = hparams["train_data_dir"]
    batch_size = hparams["batch_size"]
    lr = hparams["lr"]
    epochs = hparams["epochs"]
    train_dataset = Datasetcoloritzation(train_data_dir,device=device, training=True, image_size=256)
    train_dataset = Subset(train_dataset, range(10)) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    generator = VQResNetUNet(num_embeddings=2048, commitment_cost=0.25, out_channels_decoder=512).to(device)
    print(f"Number of trainable parameters: {count_parameters(generator)}")
    generator.apply(lambda m: normal_init(m))
    optimizer = optim.Adam(generator.parameters(), lr=lr)
    criterion = nn.L1Loss()
    pretrain_generator(generator, train_loader, optimizer, criterion, device, epochs=epochs)
    torch.save(generator.state_dict(), "pretrained_vq_GAN_test.pt")
    print(f"DONE")

 