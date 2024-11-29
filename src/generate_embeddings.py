import os
# import cv2
import argparse
import logging

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision
# from torchvision import transforms
# from torchvision.transforms import v2
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

import project_utils as utl
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


# class CapstoneDataset(Dataset):
#     def __init__(self, metadata_file, image_dir, bit_depth=8, ds_type="train", transforms=None, target_transforms=None):
#         df = pd.read_csv(metadata_file)

#         moas = df["Image_Metadata_MoA"].unique()
#         moas = moas[~pd.isnull(moas)]

#         self.class_dir = dict(zip(np.sort(moas), range(len(moas))))
#         self.classes = self.class_dir.keys()

#         df_train, df_val, df_test = utl.create_train_test_val_sets(df)
#         if ds_type == "train":
#             self.metadata = df_train
#         elif ds_type == "val":
#             self.metadata = df_val
#         elif ds_type == "test":
#             self.metadata = df_test
#         elif ds_type == "full":
#             self.metadata = df[df["Image_Metadata_MoA"].notna()]
#         else:
#             self.metadata = df_test

#         self.image_dir = image_dir
#         self.transforms = transforms
#         self.target_transforms = target_transforms
#         self.bit_depth = bit_depth

#     def __len__(self):
#         return len(self.metadata)

#     def __getitem__(self, idx):
#         img_dapi_path = os.path.join(
#             self.image_dir, self.metadata.iloc[idx, 8], self.metadata.iloc[idx, 2]
#         )
#         img_tubulin_path = os.path.join(
#             self.image_dir, self.metadata.iloc[idx, 8], self.metadata.iloc[idx, 4]
#         )
#         img_actin_path = os.path.join(
#             self.image_dir, self.metadata.iloc[idx, 8], self.metadata.iloc[idx, 6]
#         )

#         image = None
#         if self.bit_depth == 8:
#             img_dapi = cv2.imread(img_dapi_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
#             img_tubulin = cv2.imread(img_tubulin_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
#             img_actin = cv2.imread(img_actin_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

#             image = (cv2.merge([img_dapi, img_tubulin, img_actin]))
#             image /= 255.0
#         elif self.bit_depth == 16:
#             img_dapi = cv2.imread(img_dapi_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
#             img_tubulin = cv2.imread(img_tubulin_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
#             img_actin = cv2.imread(img_actin_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

#             image = (cv2.merge([img_dapi, img_tubulin, img_actin]))
#             image /= 65535.0
#         else:
#             img_dapi = cv2.imread(img_dapi_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
#             img_tubulin = cv2.imread(img_tubulin_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
#             img_actin = cv2.imread(img_actin_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

#             image = (cv2.merge([img_dapi, img_tubulin, img_actin]))
#             image /= 255.0

#         if self.transforms:
#             image = self.transforms(image)

#         # label = self.class_dir[self.metadata.iloc[idx, 13]]

#         return image, -9


def load_model(path, device, n_classes=13, original_model=False):
    pt_model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    
    if original_model:
        num_ftrs = pt_model.fc.in_features
        pt_model.fc = torch.nn.Linear(num_ftrs, n_classes)
    else:
        if device == "cpu":
            pretrained_weights = torch.load(path, map_location=torch.device("cpu"))
        else:
            pretrained_weights = torch.load(path)
        num_ftrs = pt_model.fc.in_features
        pt_model.fc = torch.nn.Linear(num_ftrs, n_classes)
        pt_model.load_state_dict(pretrained_weights, assign=True)


    embd_node = {"avgpool": "avgpool"}

    return create_feature_extractor(pt_model, return_nodes=embd_node)


def main():
    # basic logging
    # logging.basicConfig(filename="/home/huebsma1/siads699/generate_embeddings.log",
                        # format='%(message)s',
                        # filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Use device: {device}")
    logger.debug(f"Load model: {args.model_path}")

    

    # transforms = v2.Compose([v2.ToImage(),
    #                         v2.Resize((1024, 1024)), 
    #                         v2.ToDtype(torch.float32, scale=True),
    #                         v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

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
    model.eval()

    logger.debug("Start processing images")
    images_processed = 0
    embd_vecs = np.empty((0, 2048))

    with torch.no_grad():
        for i, (images, _) in enumerate(embd_dataloader):
            images = images.to(device)
            output = model(images)
            embd_batch = output["avgpool"].detach().cpu()
            embd_vecs = np.append(embd_vecs, embd_batch.flatten(start_dim=1).numpy(), axis=0)
            images_processed += len(images)

            if i % 20 == 0:
                logger.debug(f"{str(images_processed)} images processed")

    logger.debug("Finish processing images")

    np.save(args.output_path, embd_vecs, allow_pickle=False)


if __name__ == "__main__":
    main()
