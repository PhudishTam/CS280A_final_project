from pycocotools.coco import COCO
import os 
import torchvision.transforms as T
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer
import json

class Datasetcoloritzation(Dataset):
    '''
    A class used to represent a dataset for colorization.
    '''
    def __init__(self, data_dir, annotation_file=None, image_size=512, tokenizer=None, training=True):
        self.data_dir = data_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.training = training
        
        if self.training and annotation_file is not None:
            self.coco = COCO(annotation_file)
            self.image_ids = list(self.coco.imgs.keys())
        else:
            self.image_list = os.listdir(self.data_dir)
            self.image_ids = self.image_list
            with open(annotation_file, 'r') as f:
                self.annotation = json.load(f)
        
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), T.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
    
    def __getitem__(self, idx):
        if self.training:
            img_id = self.image_ids[idx]
            img_info = self.coco.loadImgs(img_id)[0]
            img_filename = img_info['file_name']
            img_path = os.path.join(self.data_dir, img_filename)
            image = Image.open(img_path).convert('RGB')
            image_resized = self.transform(image)
            gray_image = transforms.functional.rgb_to_grayscale(image_resized)
            
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            captions = max([ann['caption'] for ann in anns], key=len).lower()
            captions = "A colorful image of " + captions
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
                max_length=128
            )
            for key in tokenized_caption:
                tokenized_caption[key] = tokenized_caption[key].squeeze(0)
        else:
            tokenized_caption = None
        
        return {
            'gray_image': gray_image,
            'color_image': image_resized,
            'caption': tokenized_caption,
            "caption_text": captions
        }
    
    def __len__(self):
        return len(self.image_ids)

if __name__ == '__main__':
    #data_dir = '../initData/MS_COCO/val_set/val2017'
    #annotation_file = '../initData/MS_COCO/training_set/annotations/captions_val2017.json'
    data_dir = '../initData/MS_COCO/test_set/test2017'
    annotation_file = '../initData/MS_COCO/test_set/annotations/captions_test2017.json'
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    dataset = Datasetcoloritzation(data_dir, annotation_file, tokenizer=tokenizer,training=False)
    
    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Iterate through the DataLoader
    for batch in dataloader:
        gray_images = batch['gray_image']
        color_images = batch['color_image']
        captions = batch['caption']
        caption_texts = batch['caption_text']
        
        print(caption_texts)