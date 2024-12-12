import sys
sys.path.append("/accounts/grad/phudish_p/CS280A_final_project/src")
from pre_process_data.dataset import Datasetcoloritzation
#from model.GAN import Generator, Generator_Unet, Discriminator
from model.GAN_text import Generator_Unet, Discriminator
import torch
from torch.utils.data import DataLoader, Subset
from transformers import T5EncoderModel, T5Tokenizer
import torchvision
import numpy as np
from skimage import color
import cv2
import os 
import matplotlib.pyplot as plt


if __name__ == "__main__":
    print("Start loading the model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Generator().to(device)
    
    model = Generator_Unet().to(device)
    checkpoint = torch.load('/accounts/grad/phudish_p/CS280A_final_project/src/model/checkpoint_299_unet.pth')
    # checkpoint = torch.load('/accounts/grad/phudish_p/CS280A_final_project/model_saved/GAN_experiment_4_G_2.pt')
    # checkpoint_d = torch.load('/accounts/grad/phudish_p/CS280A_final_project/model_saved/GAN_experiment_4_D_2.pt')
    print(checkpoint.keys())   
    # print(checkpoint["epoch"])
    # print(checkpoint["train_generator_losses"])
    # print(checkpoint["train_generator_bce_losses"])
    # print(checkpoint["train_generator_l1_losses"])
    # print(checkpoint["test_generator_losses"])
    # print(checkpoint["test_generator_bce_losses"])
    # print(checkpoint["test_generator_l1_losses"])
    
    
    # plt.plot(checkpoint["train_generator_losses"], label="train_generator_losses")
    # plt.legend()
    # plt.savefig("train_generator_losses.png")
    # plt.close()
    
    # plt.plot(checkpoint["train_generator_bce_losses"], label="train_generator_bce_losses")
    # plt.legend()
    # plt.savefig("train_generator_bce_losses.png")
    # plt.close()
    
    # plt.plot(checkpoint["train_generator_l1_losses"], label="train_generator_l1_losses")
    # plt.legend()
    # plt.savefig("train_generator_l1_losses.png")
    # plt.close() 
    
    # plt.plot(checkpoint["test_generator_losses"], label="test_generator_losses")
    # plt.legend()
    # plt.savefig("test_generator_losses.png")
    # plt.close()
    
    # plt.plot(checkpoint["test_generator_bce_losses"], label="test_generator_bce_losses")
    # plt.legend()
    # plt.savefig("test_generator_bce_losses.png")
    # plt.close()

    # plt.plot(checkpoint["test_generator_l1_losses"], label="test_generator_l1_losses")
    # plt.legend()
    # plt.savefig("test_generator_l1_losses.png")
    # plt.close()
    
    
    # plt.plot(checkpoint_d["test_discriminator_losses"], label="test_discriminator_losses")
    # plt.legend()
    # plt.savefig("test_discriminator_losses.png")
    # plt.close()
    
    # plt.plot(checkpoint_d["train_discriminator_losses"], label="train_discriminator_real_losses")
    # plt.legend()
    # plt.savefig("train_discriminator_real_losses.png")
     
    model.load_state_dict(checkpoint['generator_state_dict'])
    model.eval()
    
    # Dataset setup
    validation_data_dir = "/accounts/grad/phudish_p/CS280A_final_project/initData/MS_COCO/val_set/val2017"
    train_annotation_file1 = "/accounts/grad/phudish_p/CS280A_final_project/initData/MS_COCO/training_set/annotations/captions_val2017.json"
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    text_encoder = T5EncoderModel.from_pretrained("t5-small").to(device)
    validation_dataset = Datasetcoloritzation(validation_data_dir, 
                                            annotation_file1=train_annotation_file1,
                                            device=device,
                                            tokenizer=tokenizer,
                                            training=False,
                                            image_size=256)
    
    #validation_dataset = Subset(validation_dataset, range(16))
    validation_dataloader = DataLoader(validation_dataset, batch_size=16, shuffle=True)
    image_generated = 0
    real_image_count = 0

    # Create directories to save images
    fake_image_dir = "fake_images"
    real_image_dir = "real_images"
    os.makedirs(fake_image_dir, exist_ok=True)
    os.makedirs(real_image_dir, exist_ok=True)

    with torch.no_grad():
        for batch in validation_dataloader:

            # Get L and ab channels
            L = batch['l_channels'].to(device).permute(0, 3, 1, 2).contiguous()
            ab = batch['ab_channels'].to(device).permute(0, 3, 1, 2).contiguous()
            tokenized_caption = {key: value.to(device) for key, value in batch['caption'].items()}
            with torch.no_grad():
                text_output = text_encoder(**tokenized_caption)
                text_hidden_state = text_output.last_hidden_state
                attention_mask = tokenized_caption["attention_mask"]
                seq_length = attention_mask.sum(dim=1)
                last_token_position = seq_length - 1
                batch_indices = torch.arange(text_hidden_state.shape[0]).to(device)
                text_embedding = text_hidden_state[batch_indices, last_token_position, :]
                text_embedding.to(device)

            fake_ab = model(L)
            
            L = (L + 1) * 50 
            ab = ab * 150
            fake_ab = fake_ab * 150
            
            # Concatenate L and ab channels
            fake_image = torch.cat((L, fake_ab), dim=1).detach().cpu().numpy()
            real_image = torch.cat((L, ab), dim=1).detach().cpu().numpy()

            for index, img in enumerate(fake_image):
                img = img.transpose(1, 2, 0)
                img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
                cv2.imwrite(os.path.join(fake_image_dir, f"fake_image_{image_generated}.png"), (img * 255).astype(np.uint8))
                image_generated += 1
            
            for index, img in enumerate(real_image):
                img = img.transpose(1, 2, 0)
                img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
                cv2.imwrite(os.path.join(real_image_dir, f"real_image_{real_image_count}.png"), (img * 255).astype(np.uint8))
                real_image_count += 1