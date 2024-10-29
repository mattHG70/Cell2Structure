# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 09:36:35 2024

@author: GILLEFL1
"""

import torch
import pickle
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import multiprocessing
from PIL import Image



# Load the pre-trained Inception V3 model
model_raw = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
model_raw.eval()


#freeze all parameters
for param in model_raw.parameters():
    param.requires_grad = False

#replace the last fc layer
num_ftrs = model_raw.fc.in_features
model_raw.fc = torch.nn.Linear(num_ftrs, 13) 

#set parameters of last fc open for fine tunning
for param in model_raw.fc.parameters():
    param.requires_grad = True
    
class ATDImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        
        self.image_paths = []
        for cls in self.classes:
            a_path = os.path.join(root_dir, cls, 'actin')
            t_path = os.path.join(root_dir, cls, 'tubulin')
            d_path = os.path.join(root_dir, cls, 'dapi')
            for img_name in os.listdir(a_path):
                self.image_paths.append((os.path.join(a_path, img_name),
                                         os.path.join(t_path, img_name),
                                         os.path.join(d_path, img_name),
                                         cls))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        a_path, t_path, d_path, cls = self.image_paths[idx]
        a_img = Image.open(a_path)
        t_img = Image.open(t_path)
        d_img = Image.open(d_path)

        # Normalize pixel values from 0-65535 to 0-255
        a_img = a_img.point(lambda p: p * (255.0 / 65535.0))
        t_img = t_img.point(lambda p: p * (255.0 / 65535.0))
        d_img = d_img.point(lambda p: p * (255.0 / 65535.0))

        # Convert images to grayscale
        a_img = a_img.convert('L')
        t_img = t_img.convert('L')
        d_img = d_img.convert('L')
        
        img = Image.merge('RGB', (a_img, t_img, d_img))
        if self.transform:
            img = self.transform(img)
            
        return img, self.classes.index(cls)

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])


dataset_train = ATDImageDataset(root_dir='images/sorted_reduced/train', transform = transform)
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

dataset_test = ATDImageDataset(root_dir='images/sorted_reduced/test', transform = transform)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)


def normalize_dataset(dataloader, path, mean = None, std = None):
    if mean is None:
        # Initialize variables to store the sum and std of pixel values
        mean = 0.0
        std = 0.0
        num_pixels = 0

        # Mean: 9.687756374887613e-09
        # Std: 1.1472706340498462e-08

        # Iterate through the dataset
        for images, _ in dataloader:
            batch_size, num_channels, height, width = images.shape
            num_pixels += batch_size * height * width
            mean += images.mean(axis=(0, 2, 3)).sum()
            std += images.std(axis=(0, 2, 3)).sum()

        # Calculate the mean and std
        mean /= num_pixels
        std /= num_pixels

        print(f'Mean: {mean}')
        print(f'Std: {std}')

    # Updated transform with normalization
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


    # Reload dataset with normalization
    dataset = ATDImageDataset(root_dir=path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    return dataloader

dataloader_train = normalize_dataset(dataloader_train, 'images/sorted_reduced/train')

dataloader_test = normalize_dataset(dataloader_test, 'images/sorted_reduced/test')

def train_model(model, dataloader, lr, beta1):
    print('training model with lr = {}, weight decay = {}'.format(lr, momentum))
    # Define  loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr, betas=(beta1, 0.999))

    #define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #list to store losses
    losses = []

    #define the number of epochs
    num_epochs = 10

    # Training loop

    for epoch in range(num_epochs):  
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):  
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i%10 == 0:
                print('done for batch {}, epoch {}'.format(i, epoch+1))
        losses.append(running_loss / len(dataloader))
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}')

    return model, losses

model_name = 1
for lr in [0.001, 0.01, 0.1, 1]:
    for beta1 in [0.9, 0.8, 0.99]:
    
        model, losses = train_model(model_raw, dataloader_train, beta1)
        try:
            torch.save(model.state_dict(), 'models/model{}/cnn'.format(model_name))
        except:
            os.makedirs(os.path.dirname('models/model{}/cnn'.format(model_name)))
            torch.save(model.state_dict(), 'models/model{}/cnn'.format(model_name))
        with open('models/model{}/losses'.format(model_name), "wb") as fp:   #Pickling
            pickle.dump(losses, fp)
        model_name += 1

