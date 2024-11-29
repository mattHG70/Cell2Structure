import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

import torchvision.models as models


def get_new_model(n_classes=13, unfreeze_mixed=True):
    base_model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    base_model.eval()

    #freeze all parameters
    for param in base_model.parameters():
        param.requires_grad = False

    # unfreeze the last 2 mixed layers
    if unfreeze_mixed:
        for name, param in base_model.named_parameters():
            if "Mixed_7c" in name or "Mixed_7b" in name:
                param.requires_grad = True
        
    #replace the last fc layer
    num_ftrs = base_model.fc.in_features
    base_model.fc = torch.nn.Linear(num_ftrs, 13) 
    
    #set parameters of last fc open for fine tunning
    for param in base_model.fc.parameters():
        param.requires_grad = True

    return base_model


def train_model(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    running_correct = 0.0
    for inputs, labels in dataloader:  
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device
        
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        running_correct += torch.sum(preds == labels.data)
        
    accuracy = running_correct / len(dataloader.dataset)
    losses = running_loss / len(dataloader.dataset)

    return losses, accuracy


def eval_model(model, dataloader, criterion, device):
    model.eval()
    
    validation_loss = 0.0
    validation_corr = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
    
            validation_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            validation_corr += torch.sum(preds == labels.data)

        accuracy = validation_corr / len(dataloader.dataset)
        losses = validation_loss / len(dataloader.dataset)

    return losses, accuracy


def save_model(model, path, batch_size=32, n_epochs=30, learning_rate=0.001, unfreeze_mixed=1):
    # path = "/home/mhuebsch/siads699/models"
    model_name = f"capstone_model_{str(batch_size)}_{str(n_epochs)}_{str(learning_rate)}_{str(unfreeze_mixed)}.pt"
    torch.save(model.model.state_dict(), os.path.join(path, model_name))
