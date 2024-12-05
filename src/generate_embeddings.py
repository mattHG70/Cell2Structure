import argparse

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

import capstone_dataset as cds
import capstone_transforms as trn


parser = argparse.ArgumentParser(description="Generate image embedding vectors")
parser.add_argument("-model_path", 
                    type=str, 
                    required=True, 
                    help="Path to pretrained model")
parser.add_argument("-dataset_path", 
                    type=str, 
                    required=True, 
                    help="Path to detaset csv")
parser.add_argument("-image_path", 
                    type=str, 
                    required=True, 
                    help="Path to the images")
parser.add_argument("-output_path", 
                    type=str, 
                    required=True, 
                    help="Output path to store the embedding vectors")
parser.add_argument("-batch_size", 
                    type=int, 
                    required=False,
                    default=32,
                    help="Batch size for mini batch training")
parser.add_argument("-bit_depth", 
                    type=int, 
                    choices=[8, 16], 
                    required=False, 
                    help="Bit depth of the images")
parser.add_argument("-base_model", 
                    type=bool, 
                    required=False,
                    default=False,
                    help="Use base model without finetuning")
args = parser.parse_args()

"""
Module responsible for generating the image embedding vectors base on a pre-trained model.
An Inception V3 model gets created with the ImageNet default weights. The actual 
pre-trained model is than loaded from a Pytroch model's state dict.
"""
def load_model(path, device, n_classes=13, original_model=False):
    pt_model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    
    if original_model:
        num_ftrs = pt_model.fc.in_features
        pt_model.fc = nn.Linear(num_ftrs, n_classes)
    else:
        if device == "cpu":
            pretrained_weights = torch.load(path, map_location=torch.device("cpu"))
        else:
            pretrained_weights = torch.load(path)
        num_ftrs = pt_model.fc.in_features
        pt_model.fc = nn.Linear(num_ftrs, n_classes)
        pt_model.load_state_dict(pretrained_weights, assign=True)

    # Create a Pytorch feature extractor on the last average pooling layer to
    # retrieve the flat image embedding vectors
    # Code based on Pytroch doc:
    # https://pytorch.org/vision/stable/feature_extraction.html
    embd_node = {"avgpool": "avgpool"}

    return create_feature_extractor(pt_model, return_nodes=embd_node)


def main():
    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = trn.get_transform(size=1024)

    embd_dataset = cds.CapstoneDataset(args.dataset_path,
                                       args.image_path,
                                       ds_type="full",
                                       bit_depth=args.bit_depth,
                                       transforms=transforms,
                                       target_transforms=None)

    embd_dataloader = DataLoader(embd_dataset, batch_size=batch_size, shuffle=False)

    model = load_model(args.model_path, device, n_classes=len(embd_dataset.classes), original_model=args.base_model)
    model.to(device)

    # Switch model to eval mode for prediction
    model.eval()

    embd_vecs = np.empty((0, 2048))

    # Generate the image embedding vectors
    with torch.no_grad():
        for i, (images, _) in enumerate(embd_dataloader):
            images = images.to(device)
            output = model(images)
            embd_batch = output["avgpool"].detach().cpu()
            embd_vecs = np.append(embd_vecs, embd_batch.flatten(start_dim=1).numpy(), axis=0)

    np.save(args.output_path, embd_vecs, allow_pickle=False)


if __name__ == "__main__":
    main()
