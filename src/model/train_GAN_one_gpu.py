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
from transformers import T5EncoderModel, T5Tokenizer
from torch.utils.data import Subset
from GAN import Generator, Discriminator, Generator_pretrained, Generator_Unet
from torch.utils.data.distributed import DistributedSampler


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

    print(f"Saving generator checkpoint at epoch {epoch+1}")
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

    print(f"Saving discriminator checkpoint at epoch {epoch+1}")
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
        print(f"Loading generator checkpoint from {save_path_G}")
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
        

        print(f"Loading discriminator checkpoint from {save_path_D}")
        checkpoint_D = torch.load(save_path_D)
        discriminator.load_state_dict(checkpoint_D['model_state_dict'])
        optimizer_D.load_state_dict(checkpoint_D['optimizer_state_dict'])
        train_discriminator_losses = checkpoint_D.get('train_discriminator_losses', [])
        test_discriminator_losses = checkpoint_D.get('test_discriminator_losses', [])

        return epoch, train_generator_losses, test_generator_losses, train_discriminator_losses, test_discriminator_losses, train_generator_bce_losses, test_generator_bce_losses, train_generator_l1_losses, test_generator_l1_losses
    else:
        print(f"No checkpoint found")
        return 0, [], [], [], [], [], [], [], []
    
def normal_init(m, mean=0.0, std=0.02):
    if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
        nn.init.normal_(m.weight, mean=mean, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
def train_test_epochs(generator, discriminator, text_encoder_name, train_loader, test_loader, epochs, lr, device, base_save_path, pre_train_model=False, lamb=100):
    torch.autograd.set_detect_anomaly(True)
    print("Initializing training...")
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    text_encoder = T5EncoderModel.from_pretrained(text_encoder_name).to(device)

    # Attempt to load from the base_save_path
    start_epoch, train_generator_losses, test_generator_losses, train_discriminator_losses, test_discriminator_losses, train_generator_bce_losses, test_generator_bce_losses, train_generator_l1_losses, test_generator_l1_losses = load_checkpoint(generator, discriminator, optimizer_G, optimizer_D, base_save_path)
    print(f"Starting from epoch {start_epoch}")

    # Initial test evaluation
    if start_epoch == 0:
        generator.eval()
        discriminator.eval()
        initial_generator_test_loss = 0.0
        initial_generator_bce_loss = 0.0
        initial_generator_l1_loss = 0.0
        initial_discriminator_test_loss = 0.0
        print("Starting initial test evaluation...")
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Initial Testing"):
                gray_images = batch["l_channels"].to(device).permute(0, 3, 1, 2)
                gray_images_input = gray_images.clone()
                if pre_train_model:
                    gray_images = gray_images.repeat(1, 3, 1, 1)
                ab_images = batch["ab_channels"].to(device).permute(0, 3, 1, 2)
                tokenized_caption = {key: value.to(device) for key, value in batch["caption"].items()}
                text_outputs = text_encoder(**tokenized_caption)
                text_hidden_state = text_outputs.last_hidden_state
                attention_mask = tokenized_caption["attention_mask"]
                seq_length = attention_mask.sum(dim=1)
                last_token_position = seq_length - 1
                batch_indices = torch.arange(text_hidden_state.shape[0]).to(device)
                text_embedding = text_hidden_state[batch_indices, last_token_position, :]
                
                fake_ab = generator(gray_images)
                fake_prediction = discriminator(gray_images_input.detach(), fake_ab.detach())
                loss_D_fake = discriminator.discriminator_loss_fake(fake_prediction)
                
                real_prediction = discriminator(gray_images_input, ab_images)
                loss_D_real = discriminator.discriminator_loss_real(real_prediction)
                loss_D = (loss_D_fake + loss_D_real) / 2
                
                fake_prediction = discriminator(gray_images_input, fake_ab)
                generator_bce, l1_loss = generator.generator_loss(fake_prediction, fake_ab, ab_images)
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
            gray_images_input = gray_images.clone()
            if pre_train_model:
                gray_images = gray_images.repeat(1, 3, 1, 1)
            ab_images = batch["ab_channels"].to(device).permute(0, 3, 1, 2)
            tokenized_caption = {key: value.to(device) for key, value in batch["caption"].items()}
            text_outputs = text_encoder(**tokenized_caption)
            text_hidden_state = text_outputs.last_hidden_state
            attention_mask = tokenized_caption["attention_mask"]
            seq_length = attention_mask.sum(dim=1)
            last_token_position = seq_length - 1
            batch_indices = torch.arange(text_hidden_state.shape[0]).to(device)
            text_embedding = text_hidden_state[batch_indices, last_token_position, :]
            
            fake_ab = generator(gray_images)
            fake_prediction = discriminator(gray_images_input.detach(), fake_ab.detach())
            loss_D_fake = discriminator.discriminator_loss_fake(fake_prediction)
            
            real_prediction = discriminator(gray_images_input, ab_images)
            loss_D_real = discriminator.discriminator_loss_real(real_prediction)
            loss_D = (loss_D_fake + loss_D_real) / 2
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
            
            fake_prediction = discriminator(gray_images_input, fake_ab)
            generator_bce, l1_loss = generator.generator_loss(fake_prediction, fake_ab, ab_images)
            loss_G = generator_bce + lamb * l1_loss
            optimizer_G.zero_grad()
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
                gray_images_input = gray_images.clone().contiguous()
                if pre_train_model:
                    gray_images = gray_images.repeat(1, 3, 1, 1)
                ab_images = batch["ab_channels"].to(device).permute(0, 3, 1, 2).contiguous()
                tokenized_caption = {key: value.to(device) for key, value in batch["caption"].items()}
                text_outputs = text_encoder(**tokenized_caption)
                text_hidden_state = text_outputs.last_hidden_state
                attention_mask = tokenized_caption["attention_mask"]
                seq_length = attention_mask.sum(dim=1)
                last_token_position = seq_length - 1
                batch_indices = torch.arange(text_hidden_state.shape[0]).to(device)
                text_embedding = text_hidden_state[batch_indices, last_token_position, :]
                
                fake_ab = generator(gray_images).contiguous()
                fake_prediction = discriminator(gray_images_input.detach(), fake_ab.detach())
                loss_D_fake = discriminator.discriminator_loss_fake(fake_prediction)
                
                real_prediction = discriminator(gray_images_input, ab_images)
                loss_D_real = discriminator.discriminator_loss_real(real_prediction)
                loss_D = (loss_D_fake + loss_D_real) / 2
                
                fake_prediction = discriminator(gray_images_input, fake_ab)
                generator_bce, l1_loss = generator.generator_loss(fake_prediction, fake_ab, ab_images)
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
    
    return np.array(train_generator_losses),np.array(train_generator_bce_losses),np.array(train_generator_l1_losses),np.array(test_generator_losses), np.array(test_generator_bce_losses),np.array(test_generator_l1_losses),np.array(train_discriminator_losses), np.array(test_discriminator_losses)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device set to {device}")

    json_file_path = os.path.join("/accounts/grad/phudish_p/CS280A_final_project/src", "hparams", "GAN.json")
    print(f"json_file_path: {json_file_path}")
    with open(json_file_path, "r") as f:
        hparams = json.load(f)

    train_data_dir = hparams["train_data_dir"]
    train_annotation_file1 = hparams["train_annotation_file1"]
    train_annotation_file2 = hparams["train_annotation_file2"]
    test_data_dir = hparams["test_data_dir"]
    test_annotation_file = hparams["test_annotation_file"]
    tokenizer_name = hparams["tokenizer_name"]
    text_encoder_name = hparams["text_encoder_name"]
    pre_train_model = hparams["model_name"]
    batch_size = hparams["batch_size"]
    lr = hparams["lr"]
    epochs = hparams["epochs"]

    print("Creating datasets...")
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)

    train_dataset = Datasetcoloritzation(train_data_dir, annotation_file1=train_annotation_file1, annotation_file2=train_annotation_file2,
                                        device=device, tokenizer=tokenizer, training=True, image_size=256)
    test_dataset = Datasetcoloritzation(test_data_dir, annotation_file1=test_annotation_file,
                                        device=device, tokenizer=tokenizer, training=False, image_size=256)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of testing samples: {len(test_dataset)}")

    print("Creating data loaders...")
    # max_train_samples = 100
    # max_test_samples = 20
    # train_dataset = Subset(train_dataset, range(max_train_samples))
    # test_dataset = Subset(test_dataset, range(max_test_samples))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"batch_size: {batch_size}")

    t5_encoder_hidden_size = T5EncoderModel.from_pretrained(text_encoder_name).config.hidden_size

    generator = Generator_Unet()
    discriminator = Discriminator()

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    generator.apply(lambda x: normal_init(x))
    print("Initializing models...")
    print(f"Number of parameters in Generator: {count_parameters(generator)}")
    print(f"Number of parameters in Discriminator: {count_parameters(discriminator)}")

    base_save_path = hparams["save_path"]
    print("Starting training...")
    train_losses, test_losses = train_test_epochs(generator, discriminator, text_encoder_name, train_dataloader, test_dataloader, epochs, lr, device, base_save_path, pre_train_model=False)

    print("Training finished.") 