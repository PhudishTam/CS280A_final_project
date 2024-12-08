import json
import sys 
sys.path.append("/accounts/grad/phudish_p/CS280A_final_project/src")
from pre_process_data.dataset import Datasetcoloritzation
from vae import VAE
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from diffusion_transformer import DiT
from tqdm import tqdm
import os 
from transformers import CLIPTokenizer, CLIPTextModel
from torch.utils.data import DataLoader, Subset

def scale_image(image):
    return 2.0 * image - 1.0

def save_checkpoint(model, optimizer, epoch, train_losses, test_losses, save_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_losses': train_losses,
        'test_losses': test_losses
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at {save_path} at the end of epoch {epoch+1}")

def load_checkpoint(model, optimizer, save_path):
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        test_losses = checkpoint['test_losses']
        print(f"Checkpoint loaded from {save_path}")
        return epoch, train_losses, test_losses
    else:
        print(f"No checkpoint found at {save_path}")
        return 0, [], []


def train_test_epochs(model,vae_model,text_encoder_name,train_loader,test_loader,epochs,lr,device,save_path):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(train_loader), eta_min=0)
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_name).to(device)
    model.to(device)
    vae_model.to(device)
    start_epoch, train_losses, test_losses = load_checkpoint(model, optimizer, save_path) 
    if start_epoch == 0:
        model.eval()
        initial_test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Initial Testing"):
                gray_images = batch["gray_image"].repeat(1,3,1,1).to(device)
                color_images = batch["color_image"].to(device)
                gray_images = scale_image(gray_images)
                color_images = scale_image(color_images)
                # print(f"Shape of gray_images: {gray_images.shape}")
                # print(f"Shape of color_images: {color_images.shape}")
                # print(f"min of gray_images: {torch.min(gray_images)}")
                # print(f"min of color_images: {torch.min(color_images)}")
                # print(f"max of gray_images: {torch.max(gray_images)}")
                # print(f"max of color_images: {torch.max(color_images)}")
                tokenized_caption = batch["caption"]
                # make the max length to be 77
                tokenized_caption = {key: value.to(device) for key, value in tokenized_caption.items()}
                with torch.no_grad():
                    text_outputs = text_encoder(**tokenized_caption)
                    text_hidden_state = text_outputs.last_hidden_state
                    attention_mask = tokenized_caption["attention_mask"]
                    seq_length = attention_mask.sum(dim=1)
                    last_token_position = seq_length - 1
                    #print(f"last_token_position: {last_token_position}")
                    batch_indices = torch.arange(text_hidden_state.shape[0]).to(device)
                    # shape of text_embedding: (batch_size, 512)
                    text_embedding = text_hidden_state[batch_indices,last_token_position,:]
                    # print(f"Shape of text_embedding: {text_features.shape}")
        #         #print(f"Shape of text_embedding: {text_embedding.shape}")
        #        #TODO : normalize the images to be -1, 1
                with torch.no_grad():
                    z_x_prime = vae_model.encode(gray_images)
                    z_x_prime = z_x_prime / 4.225868916511535
                    z_x = vae_model.encode(color_images)
                    z_x = z_x / 4.410986113548279
                #print(f"Shape of z_x_prime: {z_x_prime.shape}")
                #print(f"Shape of z_x: {z_x.shape}")
                delta = z_x - z_x_prime
                t = torch.rand(z_x_prime.shape[0],1).to(device)
                t_add = t.clone().unsqueeze(-1).unsqueeze(-1)
                #print(f"Shape of t: {t.shape}")
                #print(f"Shape of text_embedding: {text_embedding.shape}")
                z_t = (1-t_add) * z_x_prime + t_add * z_x
                delta_hat = model(z_t, text_embedding,t, training=False)
                loss = model.loss_fn(delta_hat, delta)
                initial_test_loss += loss.item()
                
        initial_test_loss /= len(test_loader)
        test_losses.append(initial_test_loss) 
        
    for epoch in range(start_epoch, epochs):
        model.train()
        for i,batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}")):
            gray_images = batch["gray_image"].repeat(1,3,1,1).to(device)
            color_images = batch["color_image"].to(device)
            tokenized_caption = batch["caption"].to(device)
            tokenized_caption = {key: value.to(device) for key, value in tokenized_caption.items()}
            with torch.no_grad():
                text_outputs = text_encoder(**tokenized_caption)
                text_hidden_state = text_outputs.last_hidden_state
                attention_mask = tokenized_caption["attention_mask"]
                seq_length = attention_mask.sum(dim=1)
                last_token_position = seq_length - 1
                batch_indices = torch.arange(text_hidden_state.shape[0]).to(device)
                text_embedding = text_hidden_state[batch_indices,last_token_position,:]
            optimizer.zero_grad()
            with torch.no_grad():
                z_x_prime = vae_model.encode(gray_images)
                z_x_prime = z_x_prime / 4.225868916511535
                z_x = vae_model.encode(color_images)
                z_x = z_x / 4.410986113548279
            delta = z_x - z_x_prime
            t = torch.rand((z_x_prime.shape[0],1)).to(device)
            t_add = t.clone().unsqueeze(-1).unsqueeze(-1)
            z_t = (1-t_add) * z_x_prime + t_add * z_x
            delta_hat = model(z_t,text_embedding,t, training=True)
            loss = model.loss_fn(delta_hat, delta)
            loss.backward()
            optimizer.step()
            if epoch == 0 and i < 100:
                for para_group in optimizer.param_groups:
                    para_group['lr'] = lr * (i+1) / 100
            else:
                scheduler.step()
            train_losses.append(loss.item())
        for param_group in optimizer.param_groups:
            print(f"Learning rate at the end of epoch {epoch+1}: {param_group['lr']}")
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Testing Epoch {epoch+1}/{epochs}"):
                gray_images = batch["gray_image"].repeat(1,3,1,1).to(device)
                color_images = batch["color_image"].to(device)
                tokenized_caption = batch["caption"].to(device)
                tokenized_caption = {key: value.to(device) for key, value in tokenized_caption.items()}
                with torch.no_grad():
                    text_outputs = text_encoder(**tokenized_caption)
                    text_hidden_state = text_outputs.last_hidden_state
                    attention_mask = tokenized_caption["attention_mask"]
                    seq_length = attention_mask.sum(dim=1)
                    last_token_position = seq_length - 1
                    batch_indices = torch.arange(text_hidden_state.shape[0]).to(device)
                    text_embedding = text_hidden_state[batch_indices,last_token_position,:]
                with torch.no_grad():
                    z_x_prime = vae_model.encode(gray_images)
                    z_x_prime = z_x_prime / 4.225868916511535
                    z_x = vae_model.encode(color_images)
                    z_x = z_x / 4.410986113548279 
                delta = z_x - z_x_prime
                t = torch.rand((z_x_prime.shape[0],1)).to(device)
                t_add = t.clone().unsqueeze(-1).unsqueeze(-1)
                z_t = (1-t_add) * z_x_prime + t_add * z_x
                delta_hat = model(z_t,text_embedding,t, training=False)
                loss = model.loss_fn(delta_hat, delta)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {np.mean(train_losses)}, Test Loss: {test_loss}") 
        test_losses.append(test_loss)
        save_checkpoint(model, optimizer, epoch, train_losses, test_losses, save_path) 
    return np.array(train_losses), np.array(test_losses)

if __name__ == "__main__":
    json_file_path = os.path.join("/accounts/grad/phudish_p/CS280A_final_project/src", "hparams", "diffusion_transformer.json")
    print(f"json_file_path: {json_file_path}")
    with open(json_file_path, "r") as f:
        hparams = json.load(f)
    
    train_data_dir = hparams["train_data_dir"]
    train_annotation_file1 = hparams["train_annotation_file1"]
    train_annotation_file2 = hparams["train_annotation_file2"]
    test_data_dir = hparams["test_data_dir"]
    test_annotation_file = hparams["test_annotation_file"]
    device = hparams["device"]
    tokenizer_name = hparams["tokenizer"]
    epochs = hparams["epochs"]
    lr = hparams["lr"]
    vae_name = hparams["vae_name"]
    text_encoder_name = hparams["text_encoder_name"]
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
    train_dataset = Datasetcoloritzation(train_data_dir, annotation_file1=train_annotation_file1,annotation_file2=train_annotation_file2, device=device,tokenizer=tokenizer,training=True,image_size=256, max_length=77)
    test_dataset = Datasetcoloritzation(test_data_dir, annotation_file1=test_annotation_file, device=device,tokenizer=tokenizer,training=False,image_size=256, max_length=77)
    # # make the dataset to be from -1 to 1 
    # print(f"Number of training samples: {len(train_dataset)}")
    # print(f"Number of testing samples: {len(test_dataset)}")
    # max_train_samples = 4
    # train_dataset = Subset(train_dataset,range(max_train_samples))
    # max_test_samples = 4
    # test_dataset = Subset(test_dataset,range(max_test_samples))
    train_dataloader = DataLoader(train_dataset,batch_size=256, shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size=256, shuffle=True)
    vae = VAE(vae_name).to(device)
    model = DiT(input_shape=(4,32,32),patch_size=2,hidden_size=512,num_heads=8,num_layers=12,cfg_dropout_prob=0.1).to(device)
    save_path = "/accounts/grad/phudish_p/CS280A_final_project/model_saved/model_experiment_1.pt"
    train_losses, test_losses = train_test_epochs(model,vae,text_encoder_name,train_dataloader,test_dataloader,epochs,lr,device,save_path)
    