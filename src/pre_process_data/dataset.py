from pycocotools.coco import COCO
import os 
import torchvision.transforms as T
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import json
import torch 
from skimage.color import rgb2lab
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle
from scipy.spatial import cKDTree
import faiss
# class ScaleChroma:
#     def __init__(self, scale_range=(1.0, 1.2), probability=0.1):
#         self.scale_range = scale_range
#         self.probability = probability

#     def __call__(self, img):
#         if torch.rand(1).item() < self.probability:
#             # Convert image to LAB color space
#             lab_img = rgb_to_lab(img)
#             # Ensure LAB values are within valid ranges
#             lab_img[0, :, :] = torch.clamp(lab_img[0, :, :], 0, 100)
#             lab_img[1:, :, :] = torch.clamp(lab_img[1:, :, :], -128, 127)
#             # Scale chroma channels
#             scale_factor = torch.empty(1).uniform_(*self.scale_range).item()
#             lab_img[1:, :, :] *= scale_factor
#             # Convert back to RGB
#             img = lab_to_rgb(lab_img)
#             # Clip RGB values to valid range
#             img = torch.clamp(img, 0, 1)
#         return img


# # Function to convert RGB to LAB (requires additional library, e.g., skimage)

# def rgb_to_lab(img):
#     img = img.permute(1, 2, 0).numpy()  # Convert to HWC format for skimage
#     lab_img = rgb2lab(img)
#     return torch.tensor(lab_img).permute(2, 0, 1)  # Back to CHW

# def lab_to_rgb(img):
#     # Clamp LAB values to valid ranges
#     img[0, :, :] = torch.clamp(img[0, :, :], 0, 100)
#     img[1:, :, :] = torch.clamp(img[1:, :, :], -128, 127)
#     # Convert to HWC format for skimage
#     img = img.permute(1, 2, 0).numpy()
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         rgb_img = lab2rgb(img)
#     return torch.tensor(rgb_img).permute(2, 0, 1)
class Datasetcoloritzation(Dataset):
    '''
    A class used to represent a dataset for colorization.
    '''
    def __init__(self, data_dir,annotation_file1,annotation_file2=None,device=None, image_size=512, tokenizer=None, training=True, max_length=128):
        self.data_dir = data_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.training = training
        self.device = device
        self.data_types = {}
        # self.ab_bins_path = ab_bins_path
        # self.ab_bins = np.load(ab_bins_path).astype("float32")
        # self.index = faiss.IndexFlatL2(self.ab_bins.shape[1])
        # self.index.add(self.ab_bins)
        # self.global_histogram = np.zeros(self.ab_bins.shape[0])
        if self.training and annotation_file1 is not None and annotation_file2 is not None:
            self.coco = COCO(annotation_file1)
            coco_img_ids = list(self.coco.imgs.keys())
            with open(annotation_file2, 'r') as f:
                self.annotation = json.load(f)
            custom_img_ids = [img_id.split('.')[0] for img_id in self.annotation.keys()]
            self.image_ids = list(set(coco_img_ids + custom_img_ids))
            for img_id in self.image_ids:
                if img_id in self.coco.imgs:
                    self.data_types[img_id] = "coco"
                else:
                    self.data_types[img_id] = "custom"
        else:
            self.image_list = os.listdir(self.data_dir)
            self.image_ids = self.image_list
            with open(annotation_file1, 'r') as f:
                self.annotation = json.load(f)
        
        # self.transform = transforms.Compose([
        #     transforms.Resize((self.image_size, self.image_size), T.InterpolationMode.BICUBIC),
        #     transforms.RandomApply([transforms.ColorJitter(contrast=(0.8, 1.2), brightness=(0.9, 1.1))], p=0.1),
        #     transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))], p=0.1),
        #     transforms.ToTensor(),
        # ])
        # self.transform = T.Compose([
        #     T.Resize((self.image_size, self.image_size), T.InterpolationMode.BICUBIC),
        #     #T.RandomApply([T.ColorJitter(contrast=(0.8, 1.2), brightness=(0.9, 1.1))], p=0.1
        #     #T.RandomApply([T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))], p=0.1),
        #     T.ToTensor(),
        #     T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        #     #T.RandomApply([ScaleChroma(scale_range=(1.0, 1.2))], p=0.1),
        # ])
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
        
        self.max_length = max_length
        # self.nearest_neighbors = NearestNeighbors(n_neighbors=1, algorithm='auto',n_jobs=-1).fit(self.ab_bins) 
        # self.gaussian_sigma = 5
        # self.lambda_ = 0.5
        # # self.color_distribution_path = color_distribution_path
        # with open(color_distribution_path, 'rb') as f:
        #     data = pickle.load(f)
        #     self.color_distribution = data  # If you need this
        #     self.p = data['p']
        #     self.p_smoothed = data['p_smoothed']
        #     self.w = data['w']
            
    # def quantize_ab(self, ab_values):
    #     """
    #     Function to quantize the ab values to the nearest ab bin. 
    #     """
    #     ab_values = ab_values.astype("float32")
    #     distances, indices = self.nearest_neighbors.kneighbors(ab_values)
    #     # weights = np.exp(-distances**2 / (2 * self.gaussian_sigma**2))
    #     # weights /= (np.sum(weights, axis=1, keepdims=True))
    #     soft_encoded = np.zeros((ab_values.shape[0], self.ab_bins.shape[0]))
    #     # for i in range(ab_values.shape[0]):
    #     #     soft_encoded[i, indices[i]] = 1
    #     for i in range(ab_values.shape[0]):
    #         soft_encoded[i, indices[i]] = 1
    #     return soft_encoded
    
    
    # def get_v(self, soft_encoded_ab):
    #     """
    #     Get the v values for the soft encoded ab values
        
    #     Args:
    #         soft_encoded_ab: Tensor of shape (batch_size, H, W, num_bins)
    #     """ 
    #     q_star = np.argmax(soft_encoded_ab, axis=-1)
    #     v = self.w[q_star]
    #     return v
     
    def __getitem__(self, idx):
        if self.training:
            img_id = self.image_ids[idx]
            data_type = self.data_types[img_id]
            if data_type == "coco":    
                img_info = self.coco.loadImgs(img_id)[0]
                img_filename = img_info['file_name']
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anns = self.coco.loadAnns(ann_ids)
                captions = max([ann['caption'] for ann in anns], key=len).lower()
                captions = "A colorful image of " + captions
            else:
                img_filename = str(img_id) + ".jpg"
                captions = self.annotation[img_filename].capitalize()
            img_path = os.path.join(self.data_dir, img_filename)
            image = Image.open(img_path).convert('RGB')
            image_resized = self.transform(image)
            gray_image = transforms.functional.rgb_to_grayscale(image_resized)
        else:
            img_filename = self.image_ids[idx]
            img_path = os.path.join(self.data_dir, img_filename)
            image = Image.open(img_path).convert('RGB')
            image_resized = self.transform(image)
            gray_image = transforms.functional.rgb_to_grayscale(image_resized)
            captions = self.annotation.get(img_filename, "")
        
        if self.tokenizer:
            tokenized_caption = self.tokenizer(
                captions,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=self.max_length
            )
            for key in tokenized_caption:
                tokenized_caption[key] = tokenized_caption[key].squeeze(0)
        else:
            tokenized_caption = None
        
        lab_image = rgb2lab(image_resized.permute(1, 2, 0).numpy())
        l_channel = lab_image[:, :, 0:1]
        l_channel = l_channel / 50.0 - 1.0
        ab_channels = lab_image[:, :, 1:] 
        ab_channels = ab_channels / 110.0       
        #print(f"Shape of ab_channels: {ab_channels.shape}")
        # ab_flat = ab_channels.reshape(-1, 2)
        # #print(f"Shape of ab_flat: {ab_flat.shape}")
        # # quantized_ab = self.quantize_ab(ab_flat)
        # # soft_encoded_ab = quantized_ab.reshape(ab_channels.shape[0], ab_channels.shape[1], -1)
        # # get_v = self.get_v(soft_encoded_ab)
        # #print(f"Shape of quatnized_ab_image: {quatnized_ab_image.shape}")
        return {
            'gray_image': gray_image.to(self.device),
            'color_image': image_resized.to(self.device),
            'l_channels': torch.tensor(l_channel).to(self.device),
            'ab_channels': torch.tensor(ab_channels).to(self.device),
            'caption': tokenized_caption if tokenized_caption else None,
            "caption_text": captions,
            "img_path": img_path
        }
    
    def __len__(self):
        return len(self.image_ids)


# if __name__ == '__main__':
    # data_dir = "../../initData/MS_COCO/val_set/val2017"
    # annotation_file = "../../initData/MS_COCO/training_set/annotations/captions_val2017.json"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tokenizer = T5Tokenizer.from_pretrained('t5-small')
    # dataset = Datasetcoloritzation(data_dir, annotation_file, device=device,tokenizer=tokenizer,training=True,image_size=512)
    # dataloader = DataLoader(dataset,batch_size=4, shuffle=True)
    # vae_model = VAE("stabilityai/sd-vae-ft-mse",device)
    # for batch in dataloader:
    #     gray_images = batch['gray_image'].repeat(1,3,1,1)
    #     color_images = batch['color_image']
    #     captions = batch['caption']
    #     caption_texts = batch['caption_text']
    #     # save gray_images

    