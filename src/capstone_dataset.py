import os

import cv2
import pandas as pd
import numpy as np

from torch.utils.data import Dataset

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

