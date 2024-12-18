import torch
from fastai.vision.models import resnet18
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
import json 
import sys
sys.path.append("/accounts/grad/phudish_p/CS280A_final_project/src")
from pre_process_data.dataset import Datasetcoloritzation
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn

class dynamicUnet(nn.Module):
    def __init__(self, n_input, n_output, size):
        super(dynamicUnet, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.size = size
        self.base_model = resnet18(pretrained=True)
        self.body = create_body(self.base_model, n_in=n_input, cut=-2)
        self.unet = DynamicUnet(self.body, n_output, (size, size))

    def forward(self, x):
        return self.unet(x)

def pretrain_generator(generator, train_loader, optimizer,loss_function, device, epochs):
    generator.train()
    loss_history = []
    for epoch in range(epochs):
        loss_meter = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            gray_images = batch["l_channels"].to(device).permute(0, 3, 1, 2)  
            ab_images = batch["ab_channels"].to(device).permute(0, 3, 1, 2)  
            preds = generator(gray_images)
            loss = loss_function(preds, ab_images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter += loss.item()
        avg_loss = loss_meter / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, L1 Loss: {avg_loss}")

    with open("pretraining_loss_history.json", "w") as f:
        json.dump(loss_history, f)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    json_file_path = "/accounts/grad/phudish_p/CS280A_final_project/src/hparams/GAN.json"
    with open(json_file_path, "r") as f:
        hparams = json.load(f)
    train_data_dir = hparams["train_data_dir"]
    test_data_dir = hparams["test_data_dir"]
    batch_size = hparams["batch_size"]
    lr = hparams["lr"]
    epochs = hparams["epochs"]
    train_dataset = Datasetcoloritzation(
        train_data_dir, device=device, training=True, image_size=256)
    train_dataset = Subset(train_dataset, range(1))  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    generator = dynamicUnet(n_input=1, n_output=2, size=256)
    generator = generator.to(device)
    optimizer = optim.Adam(generator.parameters(), lr=lr)
    loss_function = nn.L1Loss()
    pretrain_generator(generator, train_loader, optimizer, loss_function, device, epochs=20)
    torch.save(generator.state_dict(), "pretrained_resnet.pt")
 