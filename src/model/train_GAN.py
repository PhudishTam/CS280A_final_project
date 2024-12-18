import json 
import sys
sys.path.append("/accounts/grad/phudish_p/CS280A_final_project/src")
from pre_process_data.dataset import Datasetcoloritzation
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, Subset
from GAN import Discriminator, Generator_Unet


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Modified save_checkpoint function
def save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch,
                    train_generator_losses, train_generator_bce_losses,train_generator_l1_losses,test_generator_losses, test_generator_bce_losses, test_generator_l1_losses,
                    train_discriminator_losses, test_discriminator_losses,
                    base_save_path):
    if (epoch + 1) % 2 == 1:
        save_path_G = base_save_path.replace(".pt", "_G_1.pt")
        save_path_D = base_save_path.replace(".pt", "_D_1.pt")
    else:
        save_path_G = base_save_path.replace(".pt", "_G_2.pt")
        save_path_D = base_save_path.replace(".pt", "_D_2.pt")

    checkpoint_G = {
        'model_state_dict': generator.state_dict(),
        'optimizer_state_dict': optimizer_G.state_dict(),
        'epoch': epoch,
        'train_generator_losses': train_generator_losses,
        'test_generator_losses': test_generator_losses,
        'train_generator_bce_losses': train_generator_bce_losses,
        'test_generator_bce_losses': test_generator_bce_losses,
        'train_generator_l1_losses': train_generator_l1_losses,
        'test_generator_l1_losses': test_generator_l1_losses,
    }
    torch.save(checkpoint_G, save_path_G)

    checkpoint_D = {
        'model_state_dict': discriminator.state_dict(),
        'optimizer_state_dict': optimizer_D.state_dict(),
        'epoch': epoch,
        'train_discriminator_losses': train_discriminator_losses,
        'test_discriminator_losses': test_discriminator_losses,
    }
    torch.save(checkpoint_D, save_path_D)

# Modified load_checkpoint function  
def load_checkpoint(generator, discriminator, optimizer_G, optimizer_D, base_save_path):
    save_path_G = base_save_path.replace(".pt", "_G_1.pt")
    save_path_D = base_save_path.replace(".pt", "_D_1.pt")
    if os.path.exists(save_path_G) and os.path.exists(save_path_D):
        checkpoint_G = torch.load(save_path_G)
        generator.load_state_dict(checkpoint_G['model_state_dict'])
        optimizer_G.load_state_dict(checkpoint_G['optimizer_state_dict'])
        epoch = checkpoint_G['epoch']
        train_generator_losses = checkpoint_G.get('train_generator_losses', [])
        test_generator_losses = checkpoint_G.get('test_generator_losses', [])
        train_generator_bce_losses = checkpoint_G.get('train_generator_bce_losses', [])
        test_generator_bce_losses = checkpoint_G.get('test_generator_bce_losses', [])
        train_generator_l1_losses = checkpoint_G.get('train_generator_l1_losses', [])
        test_generator_l1_losses = checkpoint_G.get('test_generator_l1_losses', [])
        

        checkpoint_D = torch.load(save_path_D)
        discriminator.load_state_dict(checkpoint_D['model_state_dict'])
        optimizer_D.load_state_dict(checkpoint_D['optimizer_state_dict'])
        train_discriminator_losses = checkpoint_D.get('train_discriminator_losses', [])
        test_discriminator_losses = checkpoint_D.get('test_discriminator_losses', [])

        return epoch, train_generator_losses, test_generator_losses, train_discriminator_losses, test_discriminator_losses, train_generator_bce_losses, test_generator_bce_losses, train_generator_l1_losses, test_generator_l1_losses
    else:
        return 0, [], [], [], [], [], [], [], []
    
def normal_init(m, mean=0.0, std=0.02):
    if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
        nn.init.normal_(m.weight, mean=mean, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
def train_test_epochs(generator, discriminator, train_loader, test_loader, epochs, lr, device, base_save_path, lamb=100):
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # load the model     
    start_epoch, train_generator_losses, test_generator_losses, train_discriminator_losses, test_discriminator_losses, train_generator_bce_losses, test_generator_bce_losses, train_generator_l1_losses, test_generator_l1_losses = load_checkpoint(generator, discriminator, optimizer_G, optimizer_D, base_save_path)

    # Initial Test Evaluation
    if start_epoch == 0:
        generator.eval()
        discriminator.eval()
        initial_generator_test_loss = 0.0
        initial_generator_bce_loss = 0.0
        initial_generator_l1_loss = 0.0
        initial_discriminator_test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Initial Testing"):
                gray_images = batch["l_channels"].to(device).permute(0, 3, 1, 2)
                ab_images = batch["ab_channels"].to(device).permute(0, 3, 1, 2)
               
                
                fake_ab = generator(gray_images)
                fake_images = torch.cat([gray_images, fake_ab], dim=1)
                fake_prediction = discriminator(fake_images.detach())
                loss_D_fake = discriminator.discriminator_loss_fake(fake_prediction)
                real_images = torch.cat([gray_images, ab_images], dim=1)
                real_prediction = discriminator(real_images)
                loss_D_real = discriminator.discriminator_loss_real(real_prediction)
                loss_D = (loss_D_fake + loss_D_real) / 2
                
                fake_images = torch.cat([gray_images, fake_ab], dim=1)
                fake_preds = discriminator(fake_images)
                generator_bce, l1_loss = generator.generator_loss(fake_preds, fake_ab, ab_images)
                loss_G = generator_bce + lamb * l1_loss
                
                  
               
                
                initial_generator_test_loss += loss_G.item()
                initial_generator_bce_loss += generator_bce.item()
                initial_generator_l1_loss += l1_loss.item()
                initial_discriminator_test_loss += loss_D.item()
                
        initial_generator_test_loss /= len(test_loader)
        initial_discriminator_test_loss /= len(test_loader)
        test_generator_losses.append(initial_generator_test_loss)
        test_generator_bce_losses.append(initial_generator_bce_loss)
        test_generator_l1_losses.append(initial_generator_l1_loss)
        test_discriminator_losses.append(initial_discriminator_test_loss)
    
        
        print(f"Initial Generator Test Loss: {initial_generator_test_loss}")
        print(f"Initial Generator BCE Test Loss: {initial_generator_bce_loss}")
        print(f"Initial Generator L1 Test Loss: {initial_generator_l1_loss}")
        print(f"Initial Discriminator Test Loss: {initial_discriminator_test_loss}")
    
    for epoch in range(start_epoch, epochs):
        generator.train()
        discriminator.train()
        print(f"Starting epoch {epoch+1}/{epochs}")
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            gray_images = batch["l_channels"].to(device).permute(0, 3, 1, 2)
            ab_images = batch["ab_channels"].to(device).permute(0, 3, 1, 2)
            
            optimizer_D.zero_grad()
            fake_ab = generator(gray_images)
            fake_images = torch.cat([gray_images, fake_ab], dim=1)
            fake_prediction = discriminator(fake_images.detach())
            loss_D_fake = discriminator.discriminator_loss_fake(fake_prediction)
            real_images = torch.cat([gray_images, ab_images], dim=1)
            real_prediction = discriminator(real_images)
            loss_D_real = discriminator.discriminator_loss_real(real_prediction)
            loss_D = (loss_D_fake + loss_D_real) / 2
            loss_D.backward()
            optimizer_D.step()
                
            optimizer_G.zero_grad()
            fake_images = torch.cat([gray_images, fake_ab], dim=1)
            fake_preds = discriminator(fake_images)
            generator_bce, l1_loss = generator.generator_loss(fake_preds, fake_ab, ab_images)
            loss_G = generator_bce + lamb * l1_loss
            loss_G.backward()
            optimizer_G.step()
                
            train_generator_losses.append(loss_G.item())
            train_generator_bce_losses.append(generator_bce.item())
            train_generator_l1_losses.append(l1_loss.item())
            train_discriminator_losses.append(loss_D.item())

        for param_group in optimizer_G.param_groups:
            print(f"Generator Learning Rate: {param_group['lr']}")
        for param_group in optimizer_D.param_groups:
            print(f"Discriminator Learning Rate: {param_group['lr']}")
        
        generator.eval()
        discriminator.eval()
        test_generator_loss = 0.0
        test_generator_bce_loss = 0.0
        test_generator_l1_loss = 0.0
        test_discriminator_loss = 0.0
        print(f"Starting test evaluation for epoch {epoch+1}/{epochs}...")
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Testing Epoch {epoch+1}/{epochs}"):
                gray_images = batch["l_channels"].to(device).permute(0, 3, 1, 2)
                ab_images = batch["ab_channels"].to(device).permute(0, 3, 1, 2).contiguous()
               
                
                fake_ab = generator(gray_images)
                fake_images = torch.cat([gray_images, fake_ab], dim=1)
                fake_prediction = discriminator(fake_images.detach())
                loss_D_fake = discriminator.discriminator_loss_fake(fake_prediction)
                real_images = torch.cat([gray_images, ab_images], dim=1)
                real_prediction = discriminator(real_images)
                loss_D_real = discriminator.discriminator_loss_real(real_prediction)
                loss_D = (loss_D_fake + loss_D_real) / 2
                
                fake_images = torch.cat([gray_images, fake_ab], dim=1)
                fake_preds = discriminator(fake_images)
                generator_bce, l1_loss = generator.generator_loss(fake_preds, fake_ab, ab_images)
                loss_G = generator_bce + lamb * l1_loss
                 
                #loss_G = generator.generator_loss(fake_prediction, fake_ab, ab_images)
                
                test_generator_loss += loss_G.item()
                test_generator_bce_loss += generator_bce.item()
                test_generator_l1_loss += l1_loss.item()
                test_discriminator_loss += loss_D.item()
            
        test_generator_loss /= len(test_loader)
        test_generator_losses.append(test_generator_loss)
        test_generator_bce_loss /= len(test_loader)
        test_generator_bce_losses.append(test_generator_bce_loss)
        test_generator_l1_loss /= len(test_loader)
        test_generator_l1_losses.append(test_generator_l1_loss)
        test_discriminator_loss /= len(test_loader)
        test_discriminator_losses.append(test_discriminator_loss)
        
        print(f"Epoch {epoch+1}/{epochs} Generator Train Loss: {np.mean(train_generator_losses)}")
        print(f"Epoch {epoch+1}/{epochs} Generator Train BCE Loss: {np.mean(train_generator_bce_losses)}")
        print(f"Epoch {epoch+1}/{epochs} Generator Train L1 Loss: {np.mean(train_generator_l1_losses)}")
        print(f"Epoch {epoch+1}/{epochs} Discriminator Train Loss: {np.mean(train_discriminator_losses)}")
        print(f"Epoch {epoch+1}/{epochs} Generator Test Loss: {test_generator_loss}")
        print(f"Epoch {epoch+1}/{epochs} Generator Test BCE Loss: {test_generator_bce_loss}")
        print(f"Epoch {epoch+1}/{epochs} Generator Test L1 Loss: {test_generator_l1_loss}")
        print(f"Epoch {epoch+1}/{epochs} Discriminator Test Loss: {test_discriminator_loss}")
        
        save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch, train_generator_losses, train_generator_bce_losses,train_generator_l1_losses,
                        test_generator_losses, test_generator_bce_losses,test_generator_l1_losses ,train_discriminator_losses, test_discriminator_losses, base_save_path)
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device set to {device}")
    json_file_path = os.path.join("/accounts/grad/phudish_p/CS280A_final_project/src", "hparams", "GAN.json")
    print(f"json_file_path: {json_file_path}")
    with open(json_file_path, "r") as f:
        hparams = json.load(f)

    train_data_dir = hparams["train_data_dir"]
    test_data_dir = hparams["test_data_dir"]
    batch_size = hparams["batch_size"]
    lr = hparams["lr"]
    epochs = hparams["epochs"]
    print("Creating datasets")
    train_dataset = Datasetcoloritzation(train_data_dir,device=device, image_size=256, training=True)
    test_dataset = Datasetcoloritzation(test_data_dir,device=device, image_size=256, training=False)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of testing samples: {len(test_dataset)}")

    #max_train_samples = 100
    max_test_samples = 2000
    #train_dataset = Subset(train_dataset, range(max_train_samples))
    test_dataset = Subset(test_dataset, range(max_test_samples))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    generator = Generator_Unet(input_channel=1, out_channel=64, kernel_size=4, stride=2, padding=1, out_channel_decoder=512)
    discriminator = Discriminator(input_channel=3, out_channel=64, kernel_size=4, stride=2, padding=1)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    generator.apply(lambda x: normal_init(x))
    discriminator.apply(lambda x: normal_init(x))
    print(f"Number of parameters in Generator: {count_parameters(generator)}")
    print(f"Number of parameters in Discriminator: {count_parameters(discriminator)}")
    base_save_path = hparams["save_path"]
    train_test_epochs(generator, discriminator, train_dataloader, test_dataloader, epochs, lr, device, base_save_path)
    print("Training finished.") 