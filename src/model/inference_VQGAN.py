import sys
sys.path.append("/accounts/grad/phudish_p/CS280A_final_project/src")
from pre_process_data.dataset import Datasetcoloritzation
from GAN_VQ import VQResNetUNet 
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from skimage import color
import cv2
import os 
import matplotlib.pyplot as plt
from GAN import Discriminator

if __name__ == "__main__":
    print("Start loading the model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    generator = VQResNetUNet(num_embeddings=2048,commitment_cost=0.25,out_channels_decoder=512).to(device)
    discriminator = Discriminator(input_channel=3,out_channel=64, kernel_size=4,stride=2,padding=1).to(device)
    checkpoint_generator = torch.load('/accounts/grad/phudish_p/CS280A_final_project/model_saved/GAN_experiment_12_G_2.pt')
    checkpoint_discriminator = torch.load('/accounts/grad/phudish_p/CS280A_final_project/model_saved/GAN_experiment_12_D_2.pt')
    
    train_generator_losses_path_vqgan = "train_generator_GAN_losses_vqgan"
    train_generator_bce_losses_path_vqgan = "train_generator_GAN_losses_vqgan"
    train_generator_l1_losses_path_vqgan = "train_generator_GAN_losses_vqgan"
    train_discriminator_losses_path_vqgan = "train_discriminator_GAN_losses_vqgan"
    train_vq_losses_path_vqgan = "train_generator_GAN_losses_vqgan"
    test_generator_losses_path_vqgan = "test_generator_GAN_losses_vqgan"
    test_generator_bce_losses_path_vqgan = "test_generator_GAN_losses_vqgan"
    test_generator_l1_losses_path_vqgan = "test_generator_GAN_losses_vqgan"
    test_discriminator_losses_path_vqgan = "test_discriminator_GAN_losses_vqgan"
    test_vq_losses_path_vqgan = "test_generator_GAN_losses_vqgan"
    
    os.makedirs(train_generator_losses_path_vqgan, exist_ok=True)
    os.makedirs(train_discriminator_losses_path_vqgan, exist_ok=True)
    os.makedirs(test_generator_losses_path_vqgan, exist_ok=True)
    os.makedirs(test_discriminator_losses_path_vqgan, exist_ok=True)
    
    # plot the train generator losses
    plt.plot(checkpoint_generator['train_generator_losses'][:151060])
    plt.xlabel("Iteration")
    plt.ylabel("Generator Loss")
    plt.title("Train Generator Losses at every batch")
    plt.savefig(f"{train_generator_losses_path_vqgan}/train_generator_losses.png")
    plt.close()
    
    # plot the train generator bce losses
    plt.plot(checkpoint_generator['train_generator_bce_losses'][:151060])
    plt.xlabel("Iteration")
    plt.ylabel("Generator BCE Loss")
    plt.title("Train Generator BCE Losses at every batch")
    plt.savefig(f"{train_generator_bce_losses_path_vqgan}/train_generator_bce_losses.png")
    plt.close()
    
    # plot the train generaotr l1 losses
    plt.plot(checkpoint_generator['train_generator_l1_losses'][:151060])
    plt.xlabel("Iteration")
    plt.ylabel("Generator L1 Loss")
    plt.title("Train Generator L1 Losses at every batch")
    plt.savefig(f"{train_generator_l1_losses_path_vqgan}/train_generator_l1_losses.png")
    plt.close()
    
    # plot the train discriminator losses
    plt.plot(checkpoint_discriminator['train_discriminator_losses'][:151060])
    plt.xlabel("Iteration")
    plt.ylabel("Discriminator Loss")
    plt.title("Train Discriminator Losses at every batch")
    plt.savefig(f"{train_discriminator_losses_path_vqgan}/train_discriminator_losses.png")
    plt.close()

    plt.plot(checkpoint_generator['train_generator_vq_losses'][:151060])
    plt.xlabel("Iteration")
    plt.ylabel("VQ Loss")
    plt.title("Train VQ Losses at every batch")
    plt.savefig(f"{train_vq_losses_path_vqgan}/train_vq_losses.png")
    plt.close()
    
    # plot the test generator losses
    plt.plot(checkpoint_generator['test_generator_losses'][1:11])
    plt.xlabel("Epoch")
    plt.ylabel("Generator Loss")
    plt.title("Test Generator Losses at every epoch")
    plt.savefig(f"{test_generator_losses_path_vqgan}/test_generator_losses.png")
    plt.close()
    
    # plot the test generator bce losses
    plt.plot(checkpoint_generator['test_generator_bce_losses'][1:11])
    plt.xlabel("Epoch")
    plt.ylabel("Generator BCE Loss")
    plt.title("Test Generator BCE Losses at every epoch")
    plt.savefig(f"{test_generator_bce_losses_path_vqgan}/test_generator_bce_losses.png")
    plt.close()
    
    # plot the test generator l1 losses
    plt.plot(checkpoint_generator['test_generator_l1_losses'])[1:11]
    plt.xlabel("Epoch")
    plt.ylabel("Generator L1 Loss")
    plt.title("Test Generator L1 Losses at every epoch")
    plt.savefig(f"{test_generator_l1_losses_path_vqgan}/test_generator_l1_losses.png")
    plt.close()
    
    # plot the test discriminator losses
    plt.plot(checkpoint_discriminator['test_discriminator_losses'][1:11])
    plt.xlabel("Epoch")
    plt.ylabel("Discriminator Loss")
    plt.title("Test Discriminator Losses at every epoch")
    plt.savefig(f"{test_discriminator_losses_path_vqgan}/test_discriminator_losses.png")
    plt.close()
    
    plt.plot(checkpoint_generator['test_generator_vq_losses'][1:11])
    plt.xlabel("Epoch")
    plt.ylabel("VQ Loss")
    plt.title("Test VQ Losses at every epoch")
    plt.savefig(f"{test_vq_losses_path_vqgan}/test_vq_losses.png")
    plt.close()
    
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()
    
    # # Dataset setup
    # validation_data_dir = "/accounts/grad/phudish_p/CS280A_final_project/initData/MS_COCO/val_set/val2017"
    # train_annotation_file1 = "/accounts/grad/phudish_p/CS280A_final_project/initData/MS_COCO/training_set/annotations/captions_val2017.json"
    # validation_dataset = Datasetcoloritzation(validation_data_dir, 
    #                                         annotation_file1=train_annotation_file1,
    #                                         device=device,
    #                                         training=False,
    #                                         image_size=256)
    
    # #validation_dataset = Subset(validation_dataset, range(16))
    # validation_dataloader = DataLoader(validation_dataset, batch_size=16, shuffle=False)
    # image_generated = 0
    # real_image_count = 0

    # # Create directories to save images
    # fake_image_dir = "fake_images_vqgan"
    # real_image_dir = "real_images"
    # os.makedirs(fake_image_dir, exist_ok=True)
    # os.makedirs(real_image_dir, exist_ok=True)

    # with torch.no_grad():
    #     for batch in validation_dataloader:

    #         # Get L and ab channels
    #         L = batch['l_channels'].to(device).permute(0, 3, 1, 2).contiguous()
    #         ab = batch['ab_channels'].to(device).permute(0, 3, 1, 2).contiguous()
    #         L_input = L.clone().repeat(1, 3, 1, 1)
            
    #         fake_ab,_ = model(L_input)
            
    #         L = (L + 1) * 50 
    #         ab = ab * 150
    #         fake_ab = fake_ab * 150

    #         # Concatenate L and ab channels
    #         fake_image = torch.cat((L, fake_ab), dim=1).detach().cpu().numpy()
    #         real_image = torch.cat((L, ab), dim=1).detach().cpu().numpy()

    #         for index, img in enumerate(fake_image):
    #             img = img.transpose(1, 2, 0)
    #             img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    #             cv2.imwrite(os.path.join(fake_image_dir, f"fake_image_{image_generated}.png"), (img * 255).astype(np.uint8))
    #             image_generated += 1
            
    #         for index, img in enumerate(real_image):
    #             img = img.transpose(1, 2, 0)
    #             img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    #             cv2.imwrite(os.path.join(real_image_dir, f"real_image_{real_image_count}.png"), (img * 255).astype(np.uint8))
    #             real_image_count += 1