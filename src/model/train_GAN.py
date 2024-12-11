import json 
import sys
sys.path.append("/accounts/grad/phudish_p/CS280A_final_project/src")
from pre_process_data.dataset import Datasetcoloritzation
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from GAN import Generator, Discriminator, Generator_pretrained, Generator_Unet
from tqdm import tqdm
import os 
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.utils.data import Subset
from transformers import T5EncoderModel, T5Tokenizer



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print(f"Rank {rank}: DDP setup complete.")

# def scale_image(image):
#     return 2.0 * image - 1.0

def save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch,
                    train_generator_losses, test_generator_losses,
                    train_discriminator_losses, test_discriminator_losses,
                    base_save_path, rank=0):
    if rank == 0:
        if (epoch + 1) % 2 == 1:
            save_path_G = base_save_path.replace(".pt", "_G_1.pt")
            save_path_D = base_save_path.replace(".pt", "_D_1.pt")
        else:
            save_path_G = base_save_path.replace(".pt", "_G_2.pt")
            save_path_D = base_save_path.replace(".pt", "_D_2.pt")

        print(f"Rank {rank}: Saving generator checkpoint at epoch {epoch+1} to {save_path_G}")
        checkpoint_G = {
            'model_state_dict': generator.module.state_dict(),
            'optimizer_state_dict': optimizer_G.state_dict(),
            'epoch': epoch,
            'train_generator_losses': train_generator_losses,
            'test_generator_losses': test_generator_losses,
        }
        torch.save(checkpoint_G, save_path_G)
        print(f"Rank {rank}: Generator checkpoint saved at {save_path_G} at the end of epoch {epoch+1}")

        print(f"Rank {rank}: Saving discriminator checkpoint at epoch {epoch+1} to {save_path_D}")
        checkpoint_D = {
            'model_state_dict': discriminator.module.state_dict(),
            'optimizer_state_dict': optimizer_D.state_dict(),
            'epoch': epoch,
            'train_discriminator_losses': train_discriminator_losses,
            'test_discriminator_losses': test_discriminator_losses,
        }
        torch.save(checkpoint_D, save_path_D)
        print(f"Rank {rank}: Discriminator checkpoint saved at {save_path_D} at the end of epoch {epoch+1}")

def load_checkpoint(generator, discriminator, optimizer_G, optimizer_D, base_save_path, rank=0):
    save_path_G = base_save_path.replace(".pt", "_G_1.pt")
    save_path_D = base_save_path.replace(".pt", "_D_1.pt")
    if os.path.exists(save_path_G) and os.path.exists(save_path_D):
        print(f"Rank {rank}: Loading generator checkpoint from {save_path_G}")
        checkpoint_G = torch.load(save_path_G, map_location='cpu')
        generator.module.load_state_dict(checkpoint_G['model_state_dict'])
        optimizer_G.load_state_dict(checkpoint_G['optimizer_state_dict'])
        epoch = checkpoint_G['epoch']
        train_generator_losses = checkpoint_G.get('train_generator_losses', [])
        test_generator_losses = checkpoint_G.get('test_generator_losses', [])
        print(f"Rank {rank}: Generator checkpoint loaded from {save_path_G}")

        print(f"Rank {rank}: Loading discriminator checkpoint from {save_path_D}")
        checkpoint_D = torch.load(save_path_D, map_location='cpu')
        discriminator.module.load_state_dict(checkpoint_D['model_state_dict'])
        optimizer_D.load_state_dict(checkpoint_D['optimizer_state_dict'])
        train_discriminator_losses = checkpoint_D.get('train_discriminator_losses', [])
        test_discriminator_losses = checkpoint_D.get('test_discriminator_losses', [])
        print(f"Rank {rank}: Discriminator checkpoint loaded from {save_path_D}")

        return epoch, train_generator_losses, test_generator_losses, train_discriminator_losses, test_discriminator_losses
    else:
        if rank == 0:
            print(f"Rank {rank}: No checkpoint found at {save_path_G} or {save_path_D}")
        return 0, [], [], [], []
    
def normal_init(m, mean=0.0, std=0.02):
    if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
        nn.init.normal_(m.weight, mean=mean, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
def train_test_epochs(generator, discriminator, text_encoder_name, train_loader, test_loader, epochs, lr, device, base_save_path, rank, world_size,pre_train_model=False):
    torch.autograd.set_detect_anomaly(True)
    print(f"Rank {rank}: Initializing training...")
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(train_loader), eta_min=0)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    text_encoder = T5EncoderModel.from_pretrained(text_encoder_name).to(device)

    # Attempt to load from the base_save_path
    start_epoch, train_generator_losses, test_generator_losses, train_discriminator_losses, test_discriminator_losses = load_checkpoint(generator, discriminator, optimizer_G, optimizer_D, base_save_path, rank=rank)
    print(f"Rank {rank}: Starting from epoch {start_epoch}")
    #Initial test evaluation
    if start_epoch == 0:
        if isinstance(test_loader.sampler, DistributedSampler):
            test_loader.sampler.set_epoch(0)
        generator.eval()
        discriminator.eval()
        initial_generator_test_loss = 0.0
        initial_discriminator_test_loss = 0.0
        print(f"Rank {rank}: Starting initial test evaluation...")
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Initial Testing", disable=(rank!=0)):
                gray_images = batch["l_channels"].to(device).permute(0,3,1,2)
                gray_images_input = gray_images.clone()
                if pre_train_model:
                    gray_images = gray_images.repeat(1,3,1,1)
                    #print(f"Here")
                    #print(f"Shape of gray_images: {gray_images.shape}")
                ab_images = batch["ab_channels"].to(device).permute(0,3,1,2)
                tokenized_caption = {key: value.to(device) for key, value in batch["caption"].items()}
                with torch.no_grad():
                    text_outputs = text_encoder(**tokenized_caption)
                    text_hidden_state = text_outputs.last_hidden_state
                    attention_mask = tokenized_caption["attention_mask"]
                    seq_length = attention_mask.sum(dim=1)
                    last_token_position = seq_length - 1
                    batch_indices = torch.arange(text_hidden_state.shape[0]).to(device)
                    text_embedding = text_hidden_state[batch_indices,last_token_position,:]
                # print(f"Shape of gray_images: {gray_images.shape}")
                # print(f"Shape of ab_images: {ab_images.shape}")
                # print(f"Min value of gray_images: {gray_images.min()}")
                # print(f"Max value of gray_images: {gray_images.max()}")
                # print(f"Min value of ab_images: {ab_images.min()}")
                # print(f"Max value of ab_images: {ab_images.max()}")
                
                fake_ab = generator(gray_images)
                fake_prediction = discriminator(gray_images_input.detach(), fake_ab.detach())
                loss_D_fake = discriminator.module.discriminator_loss_fake(fake_prediction)
                
                real_prediction = discriminator(gray_images_input, ab_images)
                loss_D_real = discriminator.module.discriminator_loss_real(real_prediction)
                loss_D = (loss_D_fake + loss_D_real) / 2
                
                fake_prediction = discriminator(gray_images_input, fake_ab)
                loss_G = generator.module.generator_loss(fake_prediction, fake_ab, ab_images)
                
                
                initial_generator_test_loss += loss_G.item()
                initial_discriminator_test_loss += loss_D.item()
                
        initial_generator_test_loss_tensor = torch.tensor([initial_generator_test_loss]).to(device)
        initial_discriminator_test_loss_tensor = torch.tensor([initial_discriminator_test_loss]).to(device)
        dist.all_reduce(initial_generator_test_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(initial_discriminator_test_loss_tensor, op=dist.ReduceOp.SUM)
        initial_generator_test_loss = initial_generator_test_loss_tensor.item() / world_size / len(test_loader)
        initial_discriminator_test_loss = initial_discriminator_test_loss_tensor.item() / world_size / len(test_loader)
        test_generator_losses.append(initial_generator_test_loss)
        test_discriminator_losses.append(initial_discriminator_test_loss)
        
        if rank == 0:
            print(f"Initial Generator Test Loss: {initial_generator_test_loss}")
            print(f"Initial Discriminator Test Loss: {initial_discriminator_test_loss}")
    
    for epoch in range(start_epoch, epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        
        generator.train()
        discriminator.train()
        print(f"Rank {rank}: Starting epoch {epoch+1}/{epochs}")
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            with torch.no_grad():
                gray_images = batch["l_channels"].to(device).permute(0,3,1,2)
                gray_images_input = gray_images.clone()
                if pre_train_model:
                    gray_images = gray_images.repeat(1,3,1,1)
                ab_images = batch["ab_channels"].to(device).permute(0,3,1,2)
                tokenized_caption = {key: value.to(device) for key, value in batch["caption"].items()}
                text_outputs = text_encoder(**tokenized_caption)
                text_hidden_state = text_outputs.last_hidden_state
                attention_mask = tokenized_caption["attention_mask"]
                seq_length = attention_mask.sum(dim=1)
                last_token_position = seq_length - 1
                batch_indices = torch.arange(text_hidden_state.shape[0]).to(device)
                text_embedding = text_hidden_state[batch_indices,last_token_position,:]
            fake_ab = generator(gray_images)
            fake_prediction = discriminator(gray_images_input.detach(), fake_ab.detach())
            loss_D_fake = discriminator.module.discriminator_loss_fake(fake_prediction)
            
            real_prediction = discriminator(gray_images_input, ab_images)
            loss_D_real = discriminator.module.discriminator_loss_real(real_prediction)
            loss_D = (loss_D_fake + loss_D_real) / 2
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
            # for name, param in discriminator.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name} grad sizes: {param.grad.size()}, strides: {param.grad.stride()}")
            
            fake_prediction = discriminator(gray_images_input, fake_ab)
            loss_G = generator.module.generator_loss(fake_prediction, fake_ab, ab_images)
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
            # for name, param in generator.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name} grad sizes: {param.grad.size()}, strides: {param.grad.stride()}")
            
            #scheduler.step()
            train_generator_losses.append(loss_G.item())
            train_discriminator_losses.append(loss_D.item())

        if rank == 0:
            for param_group in optimizer_G.param_groups:
                print(f"Generator Learning Rate: {param_group['lr']}")
            for param_group in optimizer_D.param_groups:
                print(f"Discriminator Learning Rate: {param_group['lr']}")
        
        generator.eval()
        discriminator.eval()
        test_generator_loss = 0.0
        test_discriminator_loss = 0.0
        if isinstance(test_loader.sampler, DistributedSampler):
            test_loader.sampler.set_epoch(epoch)
        print(f"Rank {rank}: Starting test evaluation for epoch {epoch+1}/{epochs}...")
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Testing Epoch {epoch+1}/{epochs}", disable=(rank!=0)):
                gray_images = batch["l_channels"].to(device).permute(0,3,1,2)
                gray_images_input = gray_images.clone().contiguous()
                if pre_train_model:
                    gray_images = gray_images.repeat(1,3,1,1)
                ab_images = batch["ab_channels"].to(device).permute(0,3,1,2).contiguous()
                tokenized_caption = {key: value.to(device) for key, value in batch["caption"].items()}
                with torch.no_grad():
                    text_outputs = text_encoder(**tokenized_caption)
                    text_hidden_state = text_outputs.last_hidden_state
                    attention_mask = tokenized_caption["attention_mask"]
                    seq_length = attention_mask.sum(dim=1)
                    last_token_position = seq_length - 1
                    batch_indices = torch.arange(text_hidden_state.shape[0]).to(device)
                    text_embedding = text_hidden_state[batch_indices,last_token_position,:]
                
                fake_ab = generator(gray_images).contiguous()
                fake_prediction = discriminator(gray_images_input.detach(), fake_ab.detach())
                loss_D_fake = discriminator.module.discriminator_loss_fake(fake_prediction)
                
                real_prediction = discriminator(gray_images_input, ab_images)
                loss_D_real = discriminator.module.discriminator_loss_real(real_prediction)
                loss_D = (loss_D_fake + loss_D_real) / 2
                
                
                fake_prediction = discriminator(gray_images_input, fake_ab)
                loss_G = generator.module.generator_loss(fake_prediction, fake_ab, ab_images)
                
                test_generator_loss += loss_G.item()
                test_discriminator_loss += loss_D.item()
            
        
        test_generator_loss_tensor = torch.tensor([test_generator_loss]).to(device)
        test_discriminator_loss_tensor = torch.tensor([test_discriminator_loss]).to(device)
        dist.all_reduce(test_generator_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(test_discriminator_loss_tensor, op=dist.ReduceOp.SUM)
        test_generator_loss = test_generator_loss_tensor.item() / world_size / len(test_loader)
        test_generator_losses.append(test_generator_loss)
        test_discriminator_loss = test_discriminator_loss_tensor.item() / world_size / len(test_loader)
        test_discriminator_losses.append(test_discriminator_loss)
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs} Generator Train Loss: {np.mean(train_generator_losses)}, test loss: {test_generator_loss}")
            print(f"Epoch {epoch+1}/{epochs} Discriminator Train Loss: {np.mean(train_discriminator_losses)}, test loss: {test_discriminator_loss}")
        
        save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch, train_generator_losses, test_generator_losses, train_discriminator_losses, test_discriminator_losses, base_save_path, rank=rank)
    
    return np.array(train_generator_losses), np.array(test_generator_losses), np.array(train_discriminator_losses), np.array(test_discriminator_losses)

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"Rank {rank}: Before DDP setup, world_size={world_size}")

    ddp_setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}")
    print(f"Rank {rank}: Device set to {device}")

    json_file_path = os.path.join("/accounts/grad/phudish_p/CS280A_final_project/src", "hparams", "GAN.json")
    if rank == 0:
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
    #epochs = hparams["epochs"]
    batch_size = hparams["batch_size"]
    lr = hparams["lr"]
    epochs = hparams["epochs"] 
    print(f"Rank {rank}: Creating datasets...")
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
    
    train_dataset = Datasetcoloritzation(train_data_dir, annotation_file1=train_annotation_file1, annotation_file2=train_annotation_file2,
                                         device=device,tokenizer=tokenizer, training=True, image_size=256)
    test_dataset = Datasetcoloritzation(test_data_dir, annotation_file1=test_annotation_file,
                                        device=device,tokenizer=tokenizer, training=False, image_size=256)

    if rank == 0:
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of testing samples: {len(test_dataset)}")

    
    print(f"Rank {rank}: Creating data loaders...")
    #Subset the dataset for testing or debugging, if desired
    max_train_samples = 80000
    max_test_samples = 2000
    train_dataset = Subset(train_dataset, range(max_train_samples))
    test_dataset = Subset(test_dataset, range(max_test_samples))
    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
    print(f"batch_size: {batch_size}")
    #epochs= int(420000 / (len(train_dataset) // batch_size))
    t5_encoder_hidden_size = T5EncoderModel.from_pretrained(text_encoder_name).config.hidden_size
    #print(f"Rank {rank}: Number of epochs: {epochs}") 
    
    #generator = Generator(input_shape=(256,256,1), patch_size=32,num_patches=64, projection_dim=1024, num_heads=4, ff_dim=256)
    #generator = Generator_pretrained(pre_train_model)
    generator = Generator_Unet()
    
    discriminator = Discriminator()
    torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator)
    torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    generator.apply(lambda x : normal_init(x))
    print(f"Rank {rank}: Initializing models...")
    if rank == 0:
        print(f"Number of parameters in Generator: {count_parameters(generator)}")
        print(f"Number of parameters in Discriminator: {count_parameters(discriminator)}")
    
    generator = DDP(generator, device_ids=[rank], output_device=rank)
    discriminator = DDP(discriminator, device_ids=[rank], output_device=rank)
    print(f"Rank {rank}: DDP model created.")

    #base_save_path = "/accounts/grad/phudish_p/CS280A_final_project/model_saved/GAN_experiment_2.pt"
    base_save_path = hparams["save_path"]
    print(f"Rank {rank}: Starting training...")
    train_losses, test_losses = train_test_epochs(generator,discriminator, text_encoder_name, train_dataloader, test_dataloader, epochs, lr, device, base_save_path, rank, world_size, pre_train_model=False)

    print(f"Rank {rank}: Training finished, destroying process group.")
    destroy_process_group() 