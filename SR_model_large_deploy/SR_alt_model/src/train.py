import torch

import matplotlib
import matplotlib.pyplot as plt
import time
import h5py
import srcnn
import torch.optim as optim
import torch.nn as nn
import numpy as np
import math
import os
import argparse

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision.utils import save_image
from datasets import get_datasets, get_dataloaders
from utils import (
    psnr, save_model, save_model_state,
    save_plot, save_validation_results
)


matplotlib.style.use('ggplot')

parser = argparse.ArgumentParser()

parser.add_argument(
    '-e', '--epochs', default = 100, type =int
)

parser.add_argument(
    '-w', '--weights', default = None
)

args = vars(parser.parse_args())

# learning parameters
#batch_size = 64 # batch size, reduce if facing OOM error
epochs = args["epochs"]
lr = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_LABEL_PATHS = '../input/t91_hr_patches'
TRAN_IMAGE_PATHS = '../input/t91_lr_patches'
VALID_LABEL_PATHS = '../input/test_hr'
VALID_IMAGE_PATHS = '../input/test_bicubic_rgb_2x'
SAVE_VALIDATION_RESULTS = True

os.makedirs('../outputs/valid_results', exist_ok=True)

print('Computation device: ', device)
model = srcnn.SRCNN().to(device)
print(model)
if args['weights'] is not None:
    print('Loading weights to resume training...')
    checkpoint = torch.load(args['weights'])
    model.load_state_dict(checkpoint['model_state_dict'])

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

dataset_train, dataset_valid = get_datasets(
    TRAN_IMAGE_PATHS, TRAIN_LABEL_PATHS,
    VALID_IMAGE_PATHS, VALID_LABEL_PATHS
)
train_loader, valid_loader = get_dataloaders(dataset_train, dataset_valid)
print(f"Training samples: {len(dataset_train)}")
print(f"Validation samples: {len(dataset_valid)}")


def train(model, dataloader):
    model.train()
    running_loss=0.0
    running_psnr=0.0
    #total=int(len(train_data)/dataloader.batch_size)):
    for bi, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        image_data = data[0].to(device)
        label = data[1].to(device)

        optimizer.zero_grad()
        outputs = model(image_data)
        loss = criterion(outputs, label)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_psnr = psnr(label, outputs)
        running_psnr += batch_psnr
    final_loss = running_loss/len(dataloader.dataset)
    #final_psnr = running_psnr/int(len(train_data)/dataloader.batch_size)
    final_psnr = running_psnr/len(dataloader)
    return final_loss, final_psnr



def validate(model, dataloader, epoch):
    model.eval()
    running_loss=0.0
    running_psnr = 0.0
    with torch.no_grad():
        for bi, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            image_data = data[0].to(device)
            label = data[1].to(device)

            outputs = model(image_data)
            loss = criterion(outputs, label)

            running_loss += loss.item()
            batch_psnr = psnr(label, outputs)
            running_psnr += batch_psnr

        #outputs = outputs.cpu()
        #save_image(outputs, f"../outputs/val_sr{epoch}.png")
            if SAVE_VALIDATION_RESULTS and (epoch % 500) ==0:
                save_validation_results(outputs, epoch, bi)

    final_loss = running_loss/len(dataloader.dataset)
    final_psnr = running_psnr/len(dataloader)
    return final_loss, final_psnr


train_loss, val_loss = [], []
train_psnr, val_psnr = [], []
start = time.time()
for epoch in range(epochs):
    print(f"Epoch {epoch+1}of {epochs}")
    train_epoch_loss, train_epoch_psnr = train(model, train_loader)
    val_epoch_loss, val_epoch_psnr = validate(model, valid_loader, epoch+1)
    print(f"Train PSNR: {train_epoch_psnr:.3f}")
    print(f"validation PSNR: {val_epoch_psnr:.3f}")
    train_loss.append(train_epoch_loss)
    train_psnr.append(train_epoch_psnr)
    val_loss.append(val_epoch_loss)
    val_psnr.append(val_epoch_psnr)

     # Save model with all information every 100 epochs. Can be used
    # resuming training.
    if (epoch+1) % 100 == 0:
        save_model(epoch, model, optimizer, criterion)
    # Save the model state dictionary only every epoch. Small size,
    # can be used for inference.
    save_model_state(model)
    # Save the PSNR and loss plots every epoch.
    save_plot(train_loss, val_loss, train_psnr, val_psnr)

end = time.time()
print(f"Finished training in: {(end-start/60):.3f} minutes")
