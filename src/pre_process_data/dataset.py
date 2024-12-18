import os 
import torchvision.transforms as T
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch 
from skimage.color import rgb2lab

class Datasetcoloritzation(Dataset):
    def __init__(self, data_dir, device=None, image_size=256, training=True):
        self.data_dir = data_dir
        self.image_size = image_size
        self.training = training
        self.device = device
        self.image_list = os.listdir(self.data_dir)
        self.image_ids = self.image_list
        
        if self.training:
            self.transform = T.Compose([
                T.Resize((self.image_size, self.image_size), T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.RandomHorizontalFlip(),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((self.image_size, self.image_size), T.InterpolationMode.BICUBIC),
                T.ToTensor(),
            ])
        
    def __getitem__(self, idx):
        img_filename = self.image_ids[idx]
        img_path = os.path.join(self.data_dir, img_filename)
        image = Image.open(img_path).convert('RGB')
        image_resized = self.transform(image)
        gray_image = transforms.functional.rgb_to_grayscale(image_resized)
        
        lab_image = rgb2lab(image_resized.permute(1, 2, 0).numpy())
        l_channel = lab_image[:, :, 0:1]
        l_channel = l_channel / 50.0 - 1.0
        ab_channels = lab_image[:, :, 1:] 
        ab_channels = ab_channels / 110.0       
        return {
            'gray_image': gray_image.to(self.device),
            'color_image': image_resized.to(self.device),
            'l_channels': torch.tensor(l_channel).to(self.device),
            'ab_channels': torch.tensor(ab_channels).to(self.device),
            "img_path": img_path
        }
    
    def __len__(self):
        return len(self.image_ids)