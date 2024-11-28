import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration



def generate_captions(image_dir, output_json, model_name, device, batch_size=512):
    assert device == "cuda", "Only want to use cuda for this task"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
 
    image_files = os.listdir(image_dir)
    captions_dict = {}
    
    for i in tqdm(range(0, len(image_files), batch_size), desc="Generating captions"):
        batch_files = image_files[i:i + batch_size]
        images = []
        for image_file in batch_files:
            img_path = os.path.join(image_dir, image_file)
            image = Image.open(img_path).convert("RGB")
            images.append(image)
        
        inputs = processor(images, ["A colorful image of"] * len(images), return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            out = model.generate(**inputs)
        
        captions = processor.batch_decode(out, skip_special_tokens=True)
        
        for image_file, caption in zip(batch_files, captions):
            captions_dict[image_file] = caption
    
    
    with open(output_json, 'w') as f:
        json.dump(captions_dict, f, indent=4)
    
    # Debugging: Confirm that the file has been written
    print(f"Captions saved to {output_json}")

if __name__ == "__main__":
    image_dir = "initData/MS_COCO/extra_train_2017/unlabeled2017"
    os.makedirs("initData/MS_COCO/extra_train_2017/annotations", exist_ok=True)
    output_json = "initData/MS_COCO/extra_train_2017/annotations/captions_extra_train_2017.json"
    model_name = "Salesforce/blip-image-captioning-large"
    device = "cuda"
    generate_captions(image_dir, output_json, model_name, device)