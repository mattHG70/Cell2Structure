import os
import re
import pandas as pd
import numpy as np

import argparse
import logging

import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.transforms import v2
import torchvision.models as models

import project_utils as utl


class CapstoneDataset(Dataset):
    def __init__(self, metadata_file, image_dir, ds_type="train", bit_depth=8, transforms=None, target_transforms=None):
        df = pd.read_csv(metadata_file)

        moas = df["Image_Metadata_MoA"].unique()
        moas = moas[~pd.isnull(moas)]

        self.class_dir = dict(zip(np.sort(moas), range(len(moas))))
        self.classes = self.class_dir.keys()

        df_train, df_val, df_test = utl.create_train_test_val_sets(df)
        if ds_type == "train":
            self.metadata = df_train
        elif ds_type == "val":
            self.metadata = df_val
        elif ds_type == "test":
            self.metadata = df_test
        elif ds_type == "full":
            self.metadata = df[df["Image_Metadata_MoA"].notna()]
        else:
            self.metadata = df_test

        self.image_dir = image_dir
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.bit_depth = bit_depth

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_dapi_path = os.path.join(
            self.image_dir, self.metadata.iloc[idx, 8], self.metadata.iloc[idx, 2]
        )
        img_tubulin_path = os.path.join(
            self.image_dir, self.metadata.iloc[idx, 8], self.metadata.iloc[idx, 4]
        )
        img_actin_path = os.path.join(
            self.image_dir, self.metadata.iloc[idx, 8], self.metadata.iloc[idx, 6]
        )

        if self.bit_depth == 8:
            img_dapi = cv2.imread(img_dapi_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            img_tubulin = cv2.imread(img_tubulin_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            img_actin = cv2.imread(img_actin_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        elif self.bit_depth == 16:
            img_dapi = cv2.imread(img_dapi_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            img_tubulin = cv2.imread(img_tubulin_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            img_actin = cv2.imread(img_actin_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        else:
            img_dapi = cv2.imread(img_dapi_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            img_tubulin = cv2.imread(img_tubulin_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            img_actin = cv2.imread(img_actin_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        image = (cv2.merge([img_dapi, img_tubulin, img_actin]))

        if self.bit_depth == 8:
            image /= 255.0
        elif self.bit_depth == 16:
            image /= 65535.0
        else:
            image /= 255.0

        if self.transforms:
            image = self.transforms(image)

        label = self.class_dir[self.metadata.iloc[idx, 13]]

        return image, label


def get_new_model(n_classes=13, unfreeze_mixed=True):
    base_model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    base_model.eval()

    #freeze all parameters
    for param in base_model.parameters():
        param.requires_grad = False

    # unfreeze the last 2 mixed layers
    if unfreeze_mixed:
        for mod in base_model.Mixed_7c.named_modules():
            if ".conv" in mod[0]:
                nn.init.xavier_uniform_(mod[1].weight)
        for mod in base_model.Mixed_7b.named_modules():
            if ".conv" in mod[0]:
                nn.init.xavier_uniform_(mod[1].weight)

        for name, param in base_model.named_parameters():
            if "Mixed_7c" in name or "Mixed_7b" in name: # or "Mixed_7a" in name:
                param.requires_grad = True

    #replace the last fc layer
    num_ftrs = base_model.fc.in_features
    base_model.fc = torch.nn.Linear(num_ftrs, n_classes) 

    #set parameters of last fc open for fine tunning
    for param in base_model.fc.parameters():
        param.requires_grad = True

    return base_model


def train_model(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    running_correct = 0.0
    total = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device

        optimizer.zero_grad()
        outputs, _ = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        total += labels.size(0)
        running_correct += (preds == labels).sum().item()

    accuracy = running_correct / total
    losses = running_loss / total

    return losses, accuracy


def eval_model(model, dataloader, criterion, device):
    model.eval()

    validation_loss = 0.0
    validation_corr = 0.0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            validation_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            validation_corr += (preds == labels).sum().item()

        accuracy = validation_corr / total
        losses = validation_loss / total

    return losses, accuracy


def save_model(model, batch_size=32, n_epochs=30, learning_rate=0.001, bit_depth=8, unfreeze_mixed=1):
    path = project_config["training"]["modeldir"]
    model_name = f"capstone_model_{str(batch_size)}_{str(n_epochs)}_{str(learning_rate)}_{str(unfreeze_mixed)}_[str(bit_depth)}.pt"
    torch.save(model.state_dict(), os.path.join(path, model_name))


default_config_file = "../project_config.toml"

parser = argparse.ArgumentParser(description="Finetuning pre-trained model")
parser.add_argument("-bit_depth", 
                    type=int, 
                    choices=[8, 16], 
                    required=False, 
                    help="Bit depth of the images")
parser.add_argument("-n_epochs", 
                    type=int, 
                    required=False,
                    default=20
                    help="Number of training epochs")
parser.add_argument("-batch_size", 
                    type=int, 
                    required=False,
                    default=32
                    help="Batch size for mini batch training")
parser.add_argument("-learning_rate", 
                    type=int, 
                    required=False,
                    default=0.0001
                    help="Learning rate")
parser.add_argument('-config', 
                    type=str, 
                    required=False, 
                    default=default_config_file,
                    help="Project configuration file (toml format)")
args = parser.parse_args()


def main():
    # load project configuration
    global project_config
    project_config = utl.load_project_conf(args.config)
    
    # default values
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    bit_depth = args.bit_depth

    img_dir = project_config["training"]["imagefiles"]
    dataset = project_config["training"]["dataset"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # according to model documentation in PyTorch
    transforms = v2.Compose([v2.ToImage(),
                            v2.Resize((1024, 1024), antialias=True), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # basic logging
    logging.basicConfig(filename=project_config["training"]["logfile"],
                        format='%(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger.debug(f"image directory: {img_dir}")

    training_data = CapstoneDataset(
        dataset, 
        img_dir, 
        ds_type="train",
        bit_depth=bit_depth,
        transforms=transforms, 
        target_transforms=None
    )

    validation_data = CapstoneDataset(
        dataset, 
        img_dir, 
        ds_type="val",
        bit_depth=bit_depth,
        transforms=transforms, 
        target_transforms=None
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

    model = get_new_model(n_classes=len(training_data.classes), unfreeze_mixed=True)

    if torch.cuda.device_count() > 1:
        logger.debug(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

    logger.debug(f"batch size = {batch_size}, learning rate = {learning_rate}, epochs = {n_epochs}, bit_depth = {bit_depth}, device = {device}")

    results =list()
    for epoch in range(n_epochs):

        train_loss, train_accuracy = train_model(model, train_dataloader, criterion, optimizer, device)
        eval_loss, eval_accuracy = eval_model(model, val_dataloader, criterion, device)

        results.append((epoch+1, train_loss, train_accuracy, eval_loss, eval_accuracy))
        logger.debug(f"Epoch [{epoch + 1}/{n_epochs}], Train Loss: {train_loss} - Train Acc: {train_accuracy}, Val Loss:{eval_loss} - Val Acc: {eval_accuracy}")

    df_results = pd.DataFrame.from_records(results, columns=["epoch", "training loss", "training accuracy", "eval loss", "eval accuracy"])
    results_path = project_config["training"]["resultdir"]
    results_fn = f"capstone_results_{str(batch_size)}_{str(n_epochs)}_{str(learning_rate)}_{str(bit_depth)}.csv"
    df_results.to_csv(os.path.join(results_path, results_fn), index=False)

    logger.debug("-----------------------------------------------------------")

    save_model(model, batch_size=batch_size, n_epochs=n_epochs, learning_rate=learning_rate, bit_depth=bit_depth, unfreeze_mixed=1)


if __name__ == "__main__":
    main()
