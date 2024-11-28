from pycocotools.coco import COCO
import os 
import torchvision.transforms as T
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer
import json

class Datasetcoloritzation(Dataset):
    def __init__(self, data_dirs, annotation_files, tokenizer=None, training=True, device=None):
        self.data_dirs = data_dirs if isinstance(data_dirs, list) else [data_dirs]
        self.annotation_files = annotation_files if isinstance(annotation_files, list) else [annotation_files]
        self.tokenizer = tokenizer
        self.training = training
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        self.image_ids = []
        self.annotation = {}
        
        if self.training:
            self.cocos = []
            for annotation_file in self.annotation_files:
                coco = COCO(annotation_file)
                self.cocos.append(coco)
                img_ids = coco.getImgIds()
                self.image_ids.extend(img_ids)
            self.image_ids = list(set(self.image_ids))  # Remove duplicates
        else:
            for annotation_file in self.annotation_files:
                with open(annotation_file, 'r') as f:
                    annotation_data = json.load(f)
                    for item in annotation_data:
                        img_filename = item['filename']
                        captions = item.get('caption', "")
                        self.image_ids.append(img_filename)
                        self.annotation[img_filename] = captions
        
    def __getitem__(self, idx):
        if self.training:
            img_id = self.image_ids[idx]
            img_info = None
            for coco, data_dir in zip(self.cocos, self.data_dirs):
                img_infos = coco.loadImgs([img_id])
                if img_infos:
                    img_info = img_infos[0]
                    img_filename = img_info['file_name']
                    img_path = os.path.join(data_dir, img_filename)
                    if os.path.exists(img_path):
                        break
            if not img_info:
                raise FileNotFoundError(f"Image ID {img_id} not found in any dataset.")
            image = Image.open(img_path).convert('RGB')
            image_resized = self.transform(image)
            gray_image = transforms.functional.rgb_to_grayscale(image_resized)
            captions = []
            for coco in self.cocos:
                ann_ids = coco.getAnnIds(imgIds=img_id)
                anns = coco.loadAnns(ann_ids)
                captions.extend([ann['caption'] for ann in anns])
            caption = max(captions, key=len).lower() if captions else ""
            caption = "A colorful image of " + caption
        else:
            img_filename = self.image_ids[idx]
            img_path = None
            for data_dir in self.data_dirs:
                path = os.path.join(data_dir, img_filename)
                if os.path.exists(path):
                    img_path = path
                    break
            if not img_path:
                raise FileNotFoundError(f"Image {img_filename} not found in any dataset.")
            image = Image.open(img_path).convert('RGB')
            image_resized = self.transform(image)
            gray_image = transforms.functional.rgb_to_grayscale(image_resized)
            caption = self.annotation.get(img_filename, "")
        
        if self.tokenizer:
            tokenized_caption = self.tokenizer(
                caption,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=128
            )
            tokenized_caption = {k: v.squeeze(0) for k, v in tokenized_caption.items()}
        else:
            tokenized_caption = None
        
        return {
            'gray_image': gray_image.to(self.device),
            'color_image': image_resized.to(self.device),
            'caption': tokenized_caption.to(self.device) if tokenized_caption else None,
            'caption_text': caption
        }
    
    def __len__(self):
        return len(self.image_ids)

if __name__ == '__main__':
    #data_dir = '../initData/MS_COCO/val_set/val2017'
    #annotation_file = '../initData/MS_COCO/training_set/annotations/captions_val2017.json'
    data_dir = '../../initData/MS_COCO/test_set/test2017'
    annotation_file = '../../initData/MS_COCO/test_set/annotations/captions_test2017.json'
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