import json
import sys
import os
sys.path.append("/accounts/grad/phudish_p/CS280A_final_project/src")
from pre_process_data.dataset import Datasetcoloritzation
import numpy as np
from vae import VAE
import torch
import torch.nn as nn

def get_1d_sincos_pos_embed_from_grid(embed_dim,pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim//2,dtype=torch.float64)
    omega /= embed_dim / 2
    #print(f"Shape of omega : {omega.shape}")
    omega = 1 / 10000**omega
    #pos = torch.tensor([pos],dtype=torch.float64)
    #print(f"pos : {pos}")
    pos = pos.reshape(-1)
    out = torch.einsum('m,d->md',pos,omega)
    emb_sin = torch.sin(out)
    #print(f"Shape of emb_sin : {emb_sin.shape}")
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin,emb_cos],axis=1)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim,grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim//2,grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim//2,grid[1])
    emb = torch.cat([emb_h,emb_w],axis=1)
    return emb

def get_2d_sincos_pos_embed(embed_dim,grid_size):
    grid_h = torch.arange(grid_size,dtype=torch.float32)
    grid_w = torch.arange(grid_size,dtype=torch.float32)
    grid = torch.meshgrid(grid_w,grid_h)
    grid = torch.stack(grid,axis=0)
    grid = grid.reshape([2,1,grid_size,grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim,grid)
    return pos_embed

def modulate(x,shift,scale):
    return x * (1+scale.unsqueeze(1)) + shift.unsqueeze(1)

class Multihead_Attention(nn.Module):
    def __init__(self, d_model , heads ):
        super(Multihead_Attention,self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_k = self.d_model // self.heads
        self.values = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model ,d_model)
        self.last_layer = nn.Linear(d_model, d_model)


    def scaled_dot_product_attention(self,values,key,query,mask = None):
        #print(query.shape)
        first_term = torch.matmul(query,key.transpose(-2,-1)) / np.sqrt(self.d_k)
        if mask is not None:
            first_term = first_term.masked_fill(mask == 0 , -1e10)
        #print(first_term)
        softmax_first = torch.softmax(first_term, dim = -1)
        attention_score = torch.matmul(softmax_first,values)
        return attention_score


    def forward(self,values,key,query, mask = None):
        batch_size = query.size(0)
        #print(query.shape)
        query = self.query(query).view(batch_size,-1,self.heads,self.d_k).transpose(1,2)
        key = self.key(key).view(batch_size,-1,self.heads,self.d_k).transpose(1,2)
        values = self.values(values).view(batch_size,-1,self.heads,self.d_k).transpose(1,2)
        attention_score = self.scaled_dot_product_attention(values,key,query,mask)
        combined_heads_attention = attention_score.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)
        last_layer = self.last_layer(combined_heads_attention)
        #print(f"Multi head shape : {last_layer.shape}")
        return last_layer

class MLP(nn.Module):
    def __init__(self,hidden_size):
        super(MLP,self).__init__()
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size,4*hidden_size),
            nn.GELU(),
            nn.Linear(4*hidden_size,hidden_size)
        )
    def forward(self,h):
        h = self.mlp(h)
        return h

class DitBlock(nn.Module):
    def __init__(self,hidden_size,num_heads):
        super(DitBlock,self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layer_1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size,6*hidden_size,bias=True)
        )
        self.layer_norm_1 = nn.LayerNorm(hidden_size,elementwise_affine=False,eps=1e-6)
        self.attention = Multihead_Attention(d_model=hidden_size,heads=num_heads)
        self.layer_norm_2 = nn.LayerNorm(hidden_size,elementwise_affine=False,eps=1e-6)
        self.mlp = MLP(hidden_size=hidden_size)

    def forward(self,x,c):
        c = self.layer_1(c)
        #print(f"Shape of c : {c.shape}")
        shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp = c.chunk(6,dim=1)
        h = modulate(self.layer_norm_1(x),shift_msa,scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attention(h,h,h)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.layer_norm_2(x),shift_mlp,scale_mlp))
        # h = self.layer_norm_1(x)
        # h = modulate(h,shift_msa,scale_msa)
        # x = x + gate_msa.unsqueeze(1) * self.attention(h,h,h)
        # h = self.layer_norm_2(x)
        # h = modulate(h,shift_mlp,scale_mlp)
        # x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x

class FinalLayer(nn.Module):
    def __init__(self,hidden_size,patch_size,out_channels):
        super(FinalLayer,self).__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.layer_1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size,2*hidden_size,bias=True)
        )
        self.layernorm_1 = nn.LayerNorm(hidden_size,elementwise_affine=False,eps=1e-6)
        self.linear_1 = nn.Linear(hidden_size,patch_size*patch_size*out_channels,bias=True)

    def forward(self,x,c):
        c = self.layer_1(c)
        shift,scale = c.chunk(2,dim=1)
        x = self.layernorm_1(x)
        x = modulate(x,shift,scale)
        x = self.linear_1(x)
        return x

class patchify_flatten(nn.Module):
    def __init__(self,P):
        super(patchify_flatten,self).__init__()
        self.P = P
        self.linear = nn.Linear(16,512)
    def forward(self,x):
        patches = x.unfold(2,self.P,self.P).unfold(3,self.P,self.P)
        _,_,H,W = x.shape
        N = (H // self.P) * (W//self.P)
        patches_unflattend = patches.contiguous().view(x.shape[0],N,-1)
        patches_unflattend = self.linear(patches_unflattend)
        return patches_unflattend

def timestep_embedding(timesteps, dim, max_period=10000):
    half_dim = dim // 2
    freqs = torch.exp(-torch.log(torch.tensor(max_period, dtype=torch.float32)) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim)
    freqs = freqs.to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    if dim % 2:
        zero_padding = torch.zeros_like(embedding[:, :1])
        embedding = torch.cat([embedding, zero_padding], dim=-1)

    return embedding

def dropout_classes(labels, dropout_prob):
    drop_ids = torch.rand(labels.shape[0], device=labels.device) < dropout_prob
    labels = torch.where(drop_ids, 10, labels)
    return labels

class upatchify(nn.Module):
    def __init__(self,image_size):
        super(upatchify,self).__init__()
        self.image_size = image_size
    def forward(self,patches):
        B,L,D = patches.shape
        C,H,W = self.image_size
        P = int((D / C) ** 0.5)
        patches = patches.view(B, H // P, W // P, C, P, P)
        patches = patches.permute(0, 3, 1, 4, 2, 5)
        image = patches.contiguous().view(B, C, H, W)
        return image

class DiT(nn.Module):
    def __init__(self,input_shape,patch_size,hidden_size,num_heads,num_layers,num_classes,cfg_dropout_prob):
        super(DiT,self).__init__()
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.cfg_dropout_prob = cfg_dropout_prob
        self.patches = patchify_flatten(patch_size)
        num_patches = (input_shape[1]//patch_size) * (input_shape[2]//patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        self.dit_block = nn.ModuleList([DitBlock(hidden_size,num_heads) for _ in range(self.num_layers)])
        for block in self.dit_block:
            nn.init.constant_(block.layer_1[-1].weight, 0)
            nn.init.constant_(block.layer_1[-1].bias, 0)
        self.final_layer = FinalLayer(hidden_size,patch_size,4)
        # nn.init.constant_(self.final_layer.layer_1[-1].weight, 0)
        # nn.init.constant_(self.final_layer.layer_1[-1].bias, 0)
        # nn.init.constant_(self.final_layer.linear_1.weight, 0)
        # nn.init.constant_(self.final_layer.linear_1.bias, 0)
        self.upatch = upatchify((4,8,8))
        #print(self.pos_embed.shape[-1])
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],int(num_patches**0.5))
        self.pos_embed.data.copy_(pos_embed.float().unsqueeze(0))
        
    def forward(self,x,text_emb,t,training=False):
        x = self.patches(x)
        #print(f"Shape of x : {x.shape}")
        #print(f"Shape of pos_embed : {self.pos_embed.shape}")
        x += self.pos_embed
        #print(f"Shape of t before : {t.shape}")
        t = timestep_embedding(t,x.shape[-1])
        #print(f"Shape of t after : {t.shape}")
        t = t.squeeze(1)
        if training:
            text_emb = dropout_text_embedding(text_emb,self.cfg_dropout_prob)     
        c = t+text_emb
        for block in self.dit_block:
            x = block(x,c)
        x = self.final_layer(x,c)
        #print(f"Shape of x before : {x.shape}")
        x = self.upatch(x)
        #print(f"Shape of x after : {x.shape}")
        return x
        
    def loss_fn(self,epsilon,epsilon_hat):
        loss = nn.MSELoss()
        output = loss(epsilon,epsilon_hat)
        return output 
    
    def DDPM_update(self,x_t,eps_hat,t,tm1):
        alpha_t = torch.cos((np.pi/2)*t)
        alpha_t_minus_1 = torch.cos((np.pi/2)*tm1)
        sigma_t = torch.sin((np.pi/2)*t)
        sigma_t_minus_1 = torch.sin((np.pi/2)*tm1)
        eta_t = (sigma_t_minus_1)/(sigma_t)*torch.sqrt(1-(alpha_t**2/alpha_t_minus_1**2))
        epsilon_t = torch.randn_like(x_t)
        x_t_minus_1 = alpha_t_minus_1*((x_t-sigma_t*eps_hat)/(alpha_t)) + torch.sqrt(sigma_t_minus_1**2-eta_t**2)*eps_hat + eta_t*epsilon_t
        #x_t_minus_1 = alpha_t_minus_1*torch.clamp((x_t-sigma_t*eps_hat)/(alpha_t),min=-20,max=20) + torch.sqrt(sigma_t_minus_1**2-eta_t**2)*eps_hat + eta_t*epsilon_t
        return x_t_minus_1

    def sample(self,num_steps,y,device):
        self.eval()
        ts = torch.linspace(1-1e-4,1e-4,num_steps+1).to(device)
        x = torch.randn(10,4,8,8).to(device)
        #x = torch.clamp(x,-3,3)
        sample = []
        with torch.no_grad():
            for i in range(num_steps):
                t = ts[i].unsqueeze(0).unsqueeze(-1)
                tm1 = ts[i+1].unsqueeze(0).unsqueeze(-1)
                # t = t.repeat(10,1)
                # tm1 = tm1.repeat(10,1)
                eps_hat = self(x,y,t)
                x = self.DDPM_update(x,eps_hat,t,tm1)
                #x = torch.clamp(x,-3,3)
                #print(f"Shape of x : {x.shape}")
                #generated_sample = x * 0.18215
                #generated_sample = x * 1.2820
        return x


if __name__ == "__main__":
    # json_file_path = os.path.join("/accounts/grad/phudish_p/CS280A_final_project/src", "hparams", "diffusion_transformer.json")
    # print(f"json_file_path: {json_file_path}")
    # with open(json_file_path, "r") as f:
    #     hparams = json.load(f)
    
    # train_data_dir = hparams["train_data_dir"]
    # train_annotation_file1 = hparams["train_annotation_file1"]
    # train_annotation_file2 = hparams["train_annotation_file2"]
    # test_data_dir = hparams["test_data_dir"]
    # test_annotation_file = hparams["test_annotation_file"]
    # device = hparams["device"]
    # tokenizer_name = hparams["tokenizer"]
    # print(f"Device: {device}")
    # print(f"Tokenizer name: {tokenizer_name}") 