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
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.utils.data import Subset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print(f"Rank {rank}: DDP setup complete.")

def scale_image(image):
    return 2.0 * image - 1.0

def save_checkpoint(model, optimizer, epoch, train_losses, test_losses, base_save_path, rank=0):
    if rank == 0:
        if (epoch + 1) % 2 == 1:
            save_path = base_save_path.replace(".pt", "_1.pt")
        else:
            save_path = base_save_path.replace(".pt", "_2.pt")

        print(f"Rank {rank}: Saving checkpoint at epoch {epoch+1} to {save_path}")
        checkpoint = {
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'train_losses': train_losses,
            'test_losses': test_losses
        }
        torch.save(checkpoint, save_path)
        print(f"Rank {rank}: Checkpoint saved at {save_path} at the end of epoch {epoch+1}")

def load_checkpoint(model, optimizer, save_path, rank=0):
    # This currently tries to load only from the base save_path.
    # Adjust this logic if you want to attempt loading from either _1.pt or _2.pt files.
    if os.path.exists(save_path):
        print(f"Rank {rank}: Loading checkpoint from {save_path}")
        checkpoint = torch.load(save_path, map_location='cpu')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        test_losses = checkpoint['test_losses']
        if rank == 0:
            print(f"Rank {rank}: Checkpoint loaded from {save_path}")
        return epoch, train_losses, test_losses
    else:
        if rank == 0:
            print(f"Rank {rank}: No checkpoint found at {save_path}")
        return 0, [], []

def train_test_epochs(model, vae_model, text_encoder_name, train_loader, test_loader, epochs, lr, device, base_save_path, rank, world_size):
    print(f"Rank {rank}: Initializing training...")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(train_loader), eta_min=0)
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_name).to(device)

    # Attempt to load from the base_save_path
    start_epoch, train_losses, test_losses = load_checkpoint(model, optimizer, base_save_path, rank=rank)
    print(f"Rank {rank}: Starting from epoch {start_epoch}")

    # Initial test evaluation
    if start_epoch == 0:
        if isinstance(test_loader.sampler, DistributedSampler):
            test_loader.sampler.set_epoch(0)
        model.eval()
        initial_test_loss = 0.0
        print(f"Rank {rank}: Starting initial test evaluation...")
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Initial Testing", disable=(rank!=0)):
                gray_images = batch["gray_image"].repeat(1,3,1,1).to(device)
                color_images = batch["color_image"].to(device)
                gray_images = scale_image(gray_images)
                color_images = scale_image(color_images)
                tokenized_caption = {key: value.to(device) for key, value in batch["caption"].items()}
                with torch.no_grad():
                    text_outputs = text_encoder(**tokenized_caption)
                    text_hidden_state = text_outputs.last_hidden_state
                    attention_mask = tokenized_caption["attention_mask"]
                    seq_length = attention_mask.sum(dim=1)
                    last_token_position = seq_length - 1
                    batch_indices = torch.arange(text_hidden_state.shape[0]).to(device)
                    text_embedding = text_hidden_state[batch_indices,last_token_position,:]

                with torch.no_grad():
                    z_x_prime = vae_model.encode(gray_images) / 4.225868916511535
                    z_x = vae_model.encode(color_images) / 4.410986113548279
                delta = z_x - z_x_prime
                t = torch.rand(z_x_prime.shape[0],1).to(device)
                t_add = t.clone().unsqueeze(-1).unsqueeze(-1)
                z_t = (1-t_add) * z_x_prime + t_add * z_x
                delta_hat = model(z_t, text_embedding, t, training=False)
                loss = model.module.loss_fn(delta_hat, delta)
                initial_test_loss += loss.item()
                
        loss_tensor = torch.tensor([initial_test_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        initial_test_loss = loss_tensor.item() / world_size
        test_losses.append(initial_test_loss)
        if rank == 0:
            print(f"Initial Test Loss: {initial_test_loss}")
        
    for epoch in range(start_epoch, epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        print(f"Rank {rank}: Starting epoch {epoch+1}/{epochs}")
        for i,batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}", disable=(rank!=0))):
            gray_images = batch["gray_image"].repeat(1,3,1,1).to(device)
            color_images = batch["color_image"].to(device)
            tokenized_caption = {key: value.to(device) for key, value in batch["caption"].items()}
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
                z_x_prime = vae_model.encode(gray_images) / 4.225868916511535
                z_x = vae_model.encode(color_images) / 4.410986113548279
            delta = z_x - z_x_prime
            t = torch.rand((z_x_prime.shape[0],1)).to(device)
            t_add = t.clone().unsqueeze(-1).unsqueeze(-1)
            z_t = (1-t_add) * z_x_prime + t_add * z_x
            delta_hat = model(z_t, text_embedding, t, training=True)
            loss = model.module.loss_fn(z_x, z_t + delta_hat)
            loss.backward()
            optimizer.step()
            if epoch == 0 and i < 100:
                for para_group in optimizer.param_groups:
                    para_group['lr'] = lr * (i+1) / 100
            else:
                scheduler.step()
            train_losses.append(loss.item())

        if rank == 0:
            for param_group in optimizer.param_groups:
                print(f"Learning rate at the end of epoch {epoch+1}: {param_group['lr']}")

        model.eval()
        test_loss = 0.0
        if isinstance(test_loader.sampler, DistributedSampler):
            test_loader.sampler.set_epoch(epoch)
        print(f"Rank {rank}: Testing epoch {epoch+1}/{epochs}...")
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Testing Epoch {epoch+1}/{epochs}", disable=(rank!=0)):
                gray_images = batch["gray_image"].repeat(1,3,1,1).to(device)
                color_images = batch["color_image"].to(device)
                tokenized_caption = {key: value.to(device) for key, value in batch["caption"].items()}
                with torch.no_grad():
                    text_outputs = text_encoder(**tokenized_caption)
                    text_hidden_state = text_outputs.last_hidden_state
                    attention_mask = tokenized_caption["attention_mask"]
                    seq_length = attention_mask.sum(dim=1)
                    last_token_position = seq_length - 1
                    batch_indices = torch.arange(text_hidden_state.shape[0]).to(device)
                    text_embedding = text_hidden_state[batch_indices,last_token_position,:]
                with torch.no_grad():
                    z_x_prime = vae_model.encode(gray_images) / 4.225868916511535
                    z_x = vae_model.encode(color_images) / 4.410986113548279 
                delta = z_x - z_x_prime
                t = torch.rand((z_x_prime.shape[0],1)).to(device)
                t_add = t.clone().unsqueeze(-1).unsqueeze(-1)
                z_t = (1-t_add) * z_x_prime + t_add * z_x
                delta_hat = model(z_t,text_embedding,t, training=False)
                loss = model.module.loss_fn(delta_hat, delta)
                test_loss += loss.item()

        test_loss_tensor = torch.tensor([test_loss], device=device)
        dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
        test_loss = test_loss_tensor.item() / world_size / len(test_loader)
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {np.mean(train_losses)}, Test Loss: {test_loss}") 

        # Save checkpoint to one of two files based on epoch parity
        save_checkpoint(model, optimizer, epoch, train_losses, test_losses, base_save_path, rank=rank)

    return np.array(train_losses), np.array(test_losses)

if __name__ == "__main__":
    # Obtain the rank and world_size from environment variables set by torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"Rank {rank}: Before DDP setup, world_size={world_size}")

    ddp_setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}")
    print(f"Rank {rank}: Device set to {device}")

    json_file_path = os.path.join("/accounts/grad/phudish_p/CS280A_final_project/src", "hparams", "diffusion_transformer.json")
    if rank == 0:
        print(f"json_file_path: {json_file_path}")
    with open(json_file_path, "r") as f:
        hparams = json.load(f)
    
    train_data_dir = hparams["train_data_dir"]
    train_annotation_file1 = hparams["train_annotation_file1"]
    train_annotation_file2 = hparams["train_annotation_file2"]
    test_data_dir = hparams["test_data_dir"]
    test_annotation_file = hparams["test_annotation_file"]
    tokenizer_name = hparams["tokenizer"]
    epochs = hparams["epochs"]
    lr = hparams["lr"]
    vae_name = hparams["vae_name"]
    text_encoder_name = hparams["text_encoder_name"]
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)

    print(f"Rank {rank}: Creating datasets...")
    train_dataset = Datasetcoloritzation(train_data_dir, annotation_file1=train_annotation_file1, annotation_file2=train_annotation_file2,
                                         device=device, tokenizer=tokenizer, training=True, image_size=256, max_length=77)
    test_dataset = Datasetcoloritzation(test_data_dir, annotation_file1=test_annotation_file,
                                        device=device, tokenizer=tokenizer, training=False, image_size=256, max_length=77)

    if rank == 0:
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of testing samples: {len(test_dataset)}")

    print(f"Rank {rank}: Creating data loaders...")
    # Subset the dataset for testing or debugging, if desired
    # max_train_samples = 4
    # max_test_samples = 4
    # train_dataset = Subset(train_dataset, range(max_train_samples))
    # test_dataset = Subset(test_dataset, range(max_test_samples))
    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=256, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=256, sampler=test_sampler)

    print(f"Rank {rank}: Initializing models...")
    vae = VAE(vae_name).to(device)
    model = DiT(input_shape=(4,32,32), patch_size=2, hidden_size=512, num_heads=8, num_layers=12, cfg_dropout_prob=0.1).to(device)
    if rank == 0:
        print(f"Number of parameters in VAE: {count_parameters(vae)}")
        print(f"Number of parameters in DiT: {count_parameters(model)}")
    model = DDP(model, device_ids=[rank], output_device=rank)
    print(f"Rank {rank}: DDP model created.")

    base_save_path = "/accounts/grad/phudish_p/CS280A_final_project/model_saved/model_experiment_1.pt"
    print(f"Rank {rank}: Starting training...")
    train_losses, test_losses = train_test_epochs(model, vae, text_encoder_name, train_dataloader, test_dataloader, epochs, lr, device, base_save_path, rank, world_size)

    print(f"Rank {rank}: Training finished, destroying process group.")
    destroy_process_group()
