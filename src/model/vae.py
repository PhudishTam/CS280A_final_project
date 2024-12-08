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
import matplotlib.pyplot as plt
from torch.utils.data import Subset


def scale_image(image):
    return 2.0 * image - 1.0
class VAE(nn.Module):
    def __init__(self, model_name):
        super(VAE, self).__init__()
        self.model_name = model_name
        self.vae = AutoencoderKL.from_pretrained(model_name)

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
        #print(f"Shape inside forward: {images.shape}")
        latent = self.encode(images)
        reconstructed_images = self.decode(latent)
        #print(f"Shape inside after decode: {reconstructed_images.shape}")
        return reconstructed_images

def test_vae(vae, train_loader, test_loader, device):
    vae.eval()
    train_losses_color = []
    train_losses_gray = []
    test_losses_color = []
    test_losses_gray = []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc=f"Calculate train"):
            # put data to device
            gray_images = batch['gray_image'].repeat(1,3,1,1).to(device)
            color_images = batch['color_image'].to(device)
            #print(f"Shape of gray_images: {gray_images.shape}")
            #print(f"Shape of color_images: {color_images.shape}")
            reconstructed_images_gray = vae(gray_images).to(device)
            loss_gray = nn.functional.mse_loss(reconstructed_images_gray, gray_images)
            train_losses_gray.append(loss_gray.item())
            reconstructed_images_color = vae(color_images).to(device)
            loss_color = nn.functional.mse_loss(reconstructed_images_color, color_images)
            train_losses_color.append(loss_color.item())
        for batch in tqdm(test_loader, desc=f"Calculate test"):
            gray_images = batch['gray_image'].repeat(1,3,1,1).to(device)
            color_images = batch['color_image'].to(device)
            reconstructed_images_gray = vae(gray_images).to(device)
            loss_gray = nn.functional.mse_loss(reconstructed_images_gray, gray_images)
            test_losses_gray.append(loss_gray.item())
            reconstructed_images_color = vae(color_images).to(device)
            loss_color = nn.functional.mse_loss(reconstructed_images_color, color_images)
            test_losses_color.append(loss_color.item())
    return np.array(train_losses_gray), np.array(train_losses_color), np.array(test_losses_gray), np.array(test_losses_color)
if __name__ == "__main__": 
    # train_data_dir = "initData/MS_COCO/training_set/train2017"
    # train_annotation_file1 = "initData/MS_COCO/training_set/annotations/captions_train2017.json"
    # train_annotation_file2 = "initData/MS_COCO/training_set/annotations/captions_extra_train_2017.json"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # tokenizer = T5Tokenizer.from_pretrained('t5-small')
    # train_dataset = Datasetcoloritzation(train_data_dir, annotation_file1=train_annotation_file1,annotation_file2=train_annotation_file2, device=device,tokenizer=tokenizer,training=True,image_size=256)
    # #max_train_samples = 4
    # #train_dataset = Subset(train_dataset,range(max_train_samples))
    # train_dataloader = DataLoader(train_dataset,batch_size=256, shuffle=True)
    # print(f"Number of training samples: {len(train_dataset)}")
    # test_data_dir = "initData/MS_COCO/test_set/test2017"
    # test_annotation_file = "initData/MS_COCO/test_set/annotations/captions_test2017.json"
    # test_dataset = Datasetcoloritzation(test_data_dir, annotation_file1=test_annotation_file, device=device,tokenizer=tokenizer,training=False,image_size=256)
    # #max_test_samples = 4
    # #test_dataset = Subset(test_dataset,range(max_test_samples))
    # test_dataloader = DataLoader(test_dataset,batch_size=256, shuffle=True)
    # print(f"Number of testing samples: {len(test_dataset)}")
    # vae = VAE("stabilityai/sd-vae-ft-mse")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Device: {device}")
    # vae = vae.to(device)
    # # if torch.cuda.device_count() > 1:
    # #     print(f"Number of GPUs: {torch.cuda.device_count()}")
    # #     vae = nn.DataParallel(vae)
    # train_losses_gray, train_losses_color, test_losses_gray, test_losses_color = test_vae(vae, train_dataloader, test_dataloader,device)
    # np.save("train_losses_gray.npy",train_losses_gray)
    # np.save("train_losses_color.npy",train_losses_color)
    # np.save("test_losses_gray.npy",test_losses_gray)
    # np.save("test_losses_color.npy",test_losses_color)
    # print(f"Mean train loss gray : {np.mean(train_losses_gray)}")
    # print(f"Mean train loss color : {np.mean(train_losses_color)}")
    # print(f"Mean test loss gray : {np.mean(test_losses_gray)}")
    # print(f"Mean test loss color : {np.mean(test_losses_color)}")
    
    validation_data_dir = "/accounts/grad/phudish_p/CS280A_final_project/initData/MS_COCO/val_set/val2017"
    train_annotation_file1 = "/accounts/grad/phudish_p/CS280A_final_project/initData/MS_COCO/training_set/annotations/captions_val2017.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    validation_dataset = Datasetcoloritzation(validation_data_dir, annotation_file1=train_annotation_file1, device=device,tokenizer=tokenizer,training=False,image_size=256)
    validation_dataloader = DataLoader(validation_dataset,batch_size=256, shuffle=True)
    print(f"Number of validation samples: {len(validation_dataset)}")
    vae = VAE("stabilityai/sd-vae-ft-mse")
    vae.to(device)
    scale_factor_gray = []
    scale_factor_color = []
    with torch.no_grad():
        for batch in tqdm(validation_dataloader, desc=f"Calculate scale factor"):
            gray_images = batch['gray_image'].repeat(1,3,1,1).to(device)
            color_images = batch['color_image'].to(device)
            gray_images = scale_image(gray_images)
            color_images = scale_image(color_images)
            encoded_gray_images = vae.encode(gray_images)
            encoded_color_images = vae.encode(color_images)
            flatten_encoded_gray_images = nn.Flatten()(encoded_gray_images)
            flatten_encoded_color_images = nn.Flatten()(encoded_color_images)
            scale_factor_gray.append(flatten_encoded_gray_images.std().item())
            scale_factor_color.append(flatten_encoded_color_images.std().item()) 
    print(f"Mean scale factor gray : {np.mean(scale_factor_gray)}")
    print(f"Mean scale factor color : {np.mean(scale_factor_color)}")
    
             
    