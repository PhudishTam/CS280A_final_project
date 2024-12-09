import sys
sys.path.append("/accounts/grad/phudish_p/CS280A_final_project/src")
from pre_process_data.dataset import Datasetcoloritzation
from transformers import T5Tokenizer, T5EncoderModel
from torch.utils.data import DataLoader
import torchvision
import torch
import torch.nn as nn
from transformers import ViTModel

class ViTColorization(nn.Module):
    def __init__(self, model_name, text_embedding_dim, num_bins=313):
        super(ViTColorization, self).__init__()
        self.ViT = ViTModel.from_pretrained(model_name)
        vit_hidden_dim = self.ViT.config.hidden_size
        self.text_projection = nn.Linear(text_embedding_dim, vit_hidden_dim)
        self.classifer_head = nn.Linear(vit_hidden_dim, num_bins)
    
    def forward(self, grayscale_images, text_embeddings):
        """
        Args:
            grayscale_images: Tensor of shape (batch_size, 1, H, W)
            text_embeddings: Tensor of shape (batch_size, text_embedding_dim)
        
        Returns:
            color_distribution_logits : Tensor of shape (batch_size, num_patches, num_bins)
        """
        # shape of vit_outputs: (batch_size, num_patches, vit_hidden_dim)
        vit_outputs = self.ViT(pixel_values=grayscale_images).last_hidden_state
        # shape of text_embeddings: (batch_size, 1, vit_hidden_dim)
        text_embeddings = self.text_projection(text_embeddings).unsqueeze(1)
        # hape of combined_embeddings: (batch_size, num_patches+1, vit_hidden_dim)
        combined_embeddings = torch.cat([text_embeddings, vit_outputs], dim=1)
        # ignore the first patch since it is the text embedding
        logits = self.classifer_head(combined_embeddings[:, 1:, :])
        return logits
        

if __name__ == "__main__":
    validation_data_dir = "/accounts/grad/phudish_p/CS280A_final_project/initData/MS_COCO/val_set/val2017"
    train_annotation_file1 = "/accounts/grad/phudish_p/CS280A_final_project/initData/MS_COCO/training_set/annotations/captions_val2017.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    validation_dataset = Datasetcoloritzation(validation_data_dir, annotation_file1=train_annotation_file1, device=device,tokenizer=tokenizer,training=False,image_size=224)
    validation_dataloader = DataLoader(validation_dataset,batch_size=256, shuffle=True)
    print(f"Number of validation samples: {len(validation_dataset)}")
    t5_encoder = T5EncoderModel.from_pretrained("t5-small").to(device)
    model_name = "google/vit-base-patch16-224-in21k"
    text_embedding_dim = t5_encoder.config.hidden_size
    print(f"Text embedding dim: {text_embedding_dim}")
    model = ViTColorization(model_name, text_embedding_dim, num_bins=313).to(device)
    for batch in validation_dataloader:
        gray_images = batch['gray_image'].repeat(1,3,1,1).to(device)
        tokenized_caption = {key: value.to(device) for key, value in batch['caption'].items()}
        with torch.no_grad():
            text_outputs = t5_encoder(**tokenized_caption)
            text_hidden_states = text_outputs.last_hidden_state
            attention_mask = tokenized_caption['attention_mask']
            seq_length = attention_mask.sum(dim=1)
            last_token_positions = seq_length - 1
            batch_indices = torch.arange(text_hidden_states.shape[0]).to(device)
            text_embeddings = text_hidden_states[batch_indices, last_token_positions, :] 
        color_distribution_logits = model(gray_images, text_embeddings)
        print(f"Shape of color_distribution_logits: {color_distribution_logits.shape}")
        break
    
        