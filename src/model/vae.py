import sys
sys.path.append("/accounts/grad/phudish_p/CS280A_final_project/src")
from pre_process_data.dataset import Datasetcoloritzation
from diffusers.models import AutoencoderKL
from transformers import T5Tokenizer
from torch.utils.data import DataLoader
import torchvision
import torch
from torch import nn
from tqdm import tqdm
import numpy as np


class VAE(nn.Module):
    def __init__(self, model_name,device):
        super(VAE, self).__init__()
        self.model_name = model_name
        self.device = device
        self.vae = AutoencoderKL.from_pretrained(model_name).to(device)

    def encode(self, images):
        with torch.no_grad():
            latent = self.vae.encode(images)
            latent_output = latent.latent_dist.sample()
        return latent_output
    
    def decode(self, latent):
        with torch.no_grad():
            images = self.vae.decode(latent).sample
        return images
    
    def forward(self, images):
        latent = self.encode(images)
        reconstructed_images = self.decode(latent)
        return reconstructed_images

def finetune_vae(vae, train_loader,test_loader, num_epochs, lr, device):
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_losses = []
    test_losses = []
    vae.to(device)
    vae.eval()
    initial_test_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Initial testing"):
            color_images = batch['color_image'].to(device)
            reconstructed_images = vae(color_images)
            loss = criterion(reconstructed_images, color_images)
            initial_test_loss += loss.item()
    initial_test_loss /= len(test_loader)       
    test_losses.append(initial_test_loss)
    
    for epoch in range(num_epochs):
        vae.train()
        for i,data in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")):
            data = data.to(device)
            optimizer.zero_grad()
            reconstructed_images = vae(data)
            loss = criterion(reconstructed_images, data)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        vae.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Testing Epoch {epoch+1}/{num_epochs}"):
                color_images = batch['color_image'].to(device)
                reconstructed_images = vae(color_images)
                loss = criterion(reconstructed_images, color_images)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
    return vae, np.array(train_losses), np.array(test_losses)
            


if __name__ == "__main__":
    # data_dir = "../../initData/MS_COCO/val_set/val2017"
    # annotation_file = "../../initData/MS_COCO/training_set/annotations/captions_val2017.json"
    # tokenizer = T5Tokenizer.from_pretrained('t5-small')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dataset = Datasetcoloritzation(data_dir, annotation_file, device=device,tokenizer=tokenizer,training=True,image_size=512)
    # dataloader = DataLoader(dataset,batch_size=4, shuffle=True)
    # vae_model = VAE("stabilityai/sd-vae-ft-mse",device)
    # for batch in dataloader:
    #     gray_images = batch['gray_image'].repeat(1,3,1,1)
    #     color_images = batch['color_image']
    #     captions = batch['caption']
    #     caption_texts = batch['caption_text']
    #     # save gray_images 
    #     latents = vae_model.encode(color_images)
    #     print(latents.shape)
    #     reconstructed_images = vae_model.decode(latents)
    #     torchvision.utils.save_image(reconstructed_images, "reconstructed_images.png")
    #     torchvision.utils.save_image(gray_images, "gray_images.png")
    #     torchvision.utils.save_image(color_images, "color_images.png")
    #     print(caption_texts)
    #     break
    # model = "CompVis/stable-diffusion-v1-4"
    # vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    # pipe = StableDiffusionPipeline.from_pretrained(model, vae=vae)
