from diffusion_transformer import DiT
import torch
from torch.utils.data import DataLoader, Subset
from pre_process_data.dataset import Datasetcoloritzation
from transformers import CLIPTokenizer, CLIPTextModel
import torchvision
import sys
from vae import VAE
sys.path.append("/accounts/grad/phudish_p/CS280A_final_project/src")

def scale_image(image):
    return 2.0 * image - 1.0
if __name__ == '__main__':
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiT(input_shape=(4,8,8), patch_size=2, hidden_size=512, num_heads=16, num_layers=28, cfg_dropout_prob=0.1).to(device)    
    checkpoint = torch.load('/accounts/grad/phudish_p/CS280A_final_project/model_saved/model_test_2.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully")
    # load the validation data 
    validation_data_dir = "/accounts/grad/phudish_p/CS280A_final_project/initData/MS_COCO/val_set/val2017"
    train_annotation_file1 = "/accounts/grad/phudish_p/CS280A_final_project/initData/MS_COCO/training_set/annotations/captions_val2017.json"
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    validation_dataset = Datasetcoloritzation(validation_data_dir, annotation_file1=train_annotation_file1, device=device,tokenizer=tokenizer,training=False,image_size=64, max_length=77)
    max_validation_samples = 8
    validation_dataset = Subset(validation_dataset,range(max_validation_samples))
    validation_dataloader = DataLoader(validation_dataset,batch_size=4, shuffle=True)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    guidance_scale = [1.6]
    color_scale = [0.8]
    print(f"Number of validation samples: {len(validation_dataset)}")
    vae = VAE("stabilityai/sd-vae-ft-mse").to(device) 
    with torch.no_grad():
        for batch in validation_dataloader:
            gray_images = batch['gray_image'].repeat(1,3,1,1).to(device)
            color_images = batch['color_image'].to(device)
            torchvision.utils.save_image(gray_images, "gray_images_test.png")
            torchvision.utils.save_image(color_images, "color_images_test.png")
            baseline_color = (gray_images + color_images) / 2  # Example baseline
            torchvision.utils.save_image(baseline_color, "baseline_output.png")
            gray_images = scale_image(gray_images)
            color_images = scale_image(color_images)
            num_steps = 256
            positive_captions = batch["caption"].to(device)
            tokenized_postive_captions = {key: value.to(device) for key, value in positive_captions.items()}
            with torch.no_grad():
                positive_text_outputs = text_encoder(**tokenized_postive_captions)
                positive_text_hidden_state = positive_text_outputs.last_hidden_state
                attention_mask_positive = tokenized_postive_captions["attention_mask"]
                seq_length_positive = attention_mask_positive.sum(dim=1)
                last_token_position_positive = seq_length_positive - 1
                batch_indices_positive = torch.arange(positive_text_hidden_state.shape[0]).to(device)
                positive_text_embedding = positive_text_hidden_state[batch_indices_positive,last_token_position_positive,:]
                
                negative_captions = [f"Remove grayscale tones from the image for sample {i}" for i in range(positive_text_hidden_state.shape[0])] 
                tokenized_negative_captions = tokenizer(negative_captions, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
                negative_text_outputs = text_encoder(**tokenized_negative_captions)
                negative_text_hidden_state = negative_text_outputs.last_hidden_state
                attention_mask_negative = tokenized_negative_captions["attention_mask"]
                seq_length_negative = attention_mask_negative.sum(dim=1)
                last_token_position_negative = seq_length_negative - 1
                batch_indices_negative = torch.arange(negative_text_hidden_state.shape[0]).to(device)
                negative_text_embedding = negative_text_hidden_state[batch_indices_negative,last_token_position_negative,:]
                
                positive_text_embedding = positive_text_embedding.to(device)
                negative_text_embedding = negative_text_embedding.to(device)
            
            for guidance in guidance_scale:
                for color in color_scale:
                    z_x = model.sample(vae, gray_images, num_steps, device, positive_text_embedding, negative_text_embedding, guidance_scale=guidance, color_scale=color)
                    z_x = z_x * 4.410986113548279
                    decoded_samples = (vae.decode(z_x))
                    decoded_samples = (decoded_samples + 1) / 2
                    print(f"Shape of decoded samples: {decoded_samples.shape}")
                    torchvision.utils.save_image(decoded_samples, f"decoded_samples_guidance_{guidance}_color_{color}.png")
                    
            