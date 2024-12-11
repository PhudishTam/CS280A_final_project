import sys
sys.path.append("/accounts/grad/phudish_p/CS280A_final_project/src")
from pre_process_data.dataset import Datasetcoloritzation
from model.GAN import Generator, Generator_Unet
import torch
from torch.utils.data import DataLoader, Subset
from transformers import T5EncoderModel, T5Tokenizer
import torchvision
import numpy as np
from skimage import color
import cv2



if __name__ == "__main__":
    print("Start loading the model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = Generator().to(device)
    model = Generator_Unet().to(device)
    checkpoint = torch.load('/accounts/grad/phudish_p/CS280A_final_project/model_saved/GAN_experiment_2_G_2.pt')
    print(checkpoint.keys())   
    print(checkpoint["epoch"])
    print(checkpoint["train_generator_losses"])
    print(checkpoint["test_generator_losses"])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Dataset setup
    validation_data_dir = "/accounts/grad/phudish_p/CS280A_final_project/initData/MS_COCO/val_set/val2017"
    train_annotation_file1 = "/accounts/grad/phudish_p/CS280A_final_project/initData/MS_COCO/training_set/annotations/captions_val2017.json"
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    validation_dataset = Datasetcoloritzation(validation_data_dir, 
                                            annotation_file1=train_annotation_file1,
                                            device=device,
                                            tokenizer=tokenizer,
                                            training=False,
                                            image_size=256)
    
    validation_dataset = Subset(validation_dataset, range(16), shuffle=Tru
    validation_dataloader = DataLoader(validation_dataset, batch_size=8, shuffle=True)
    image_generated = 0
    real_image_count = 0
    with torch.no_grad():
        for batch in validation_dataloader:
            # Get L and ab channels
            L = batch['l_channels'].to(device).permute(0, 3, 1, 2).contiguous()
            ab = batch['ab_channels'].to(device).permute(0, 3, 1, 2).contiguous()
            
            fake_ab = model(L)
            
            L = (L + 1) * 50 
            ab = ab * 150
            fake_ab = fake_ab * 150
            
            # Concatenate L and ab channels
            fake_image = torch.cat((L, fake_ab), dim=1).detach().cpu().numpy()
            real_image = torch.cat((L, ab), dim=1).detach().cpu().numpy()

            
            for index,img in enumerate(fake_image):
                img = img.transpose(1, 2, 0)
                img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
                cv2.imwrite(f"fake_image_{image_generated}.png", (img * 255).astype(np.uint8))
                image_generated += 1
            
            for index,img in enumerate(real_image):
                img = img.transpose(1, 2, 0)
                img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
                cv2.imwrite(f"real_image_{real_image_count}.png", (img * 255).astype(np.uint8))
                real_image_count += 1
                
            
            