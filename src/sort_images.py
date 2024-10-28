import os
import shutil
import argparse

import pandas as pd
import numpy as np

import project_utils as utl


default_config_file = "../project_config.toml"
default_image_dir = "/scratch/siads699f24_class_root/siads699f24_class/mhuebsch/images"
default_data_file = "../data/processed/BBBC021_final_original_dataset.csv"
default_max_images = 100


parser = argparse.ArgumentParser(prog="sort_images.py", 
                                 description="Sort images into train and test sets for finetuning")
parser.add_argument("-max_images", 
                    type=int, 
                    required=False, 
                    default=default_max_images,
                    help="Max images in test and train set")
parser.add_argument("-image_dir", 
                    type=str, 
                    required=False, 
                    default=default_image_dir,
                    help="Image directory on scratch or temp storage")
parser.add_argument("-datafile", 
                    type=str, 
                    required=False, 
                    default=default_data_file,
                    help="Data file containing compounds, SMILES and image metadata")
parser.add_argument("-config", 
                    type=str, 
                    required=False, 
                    default=default_config_file,
                    help="Project configuration file (toml format)")
args = parser.parse_args()


def sort_images(df, source_path, destination_path, max_img_per_moa=None):

    #first we iterate through each MoA
    for moa in df.Image_Metadata_MoA.dropna().unique():
        df_filtered = df[df.Image_Metadata_MoA == moa]

        if max_img_per_moa is None:
            loop_increment = 1 #if we do not want to limit the number of images per MoA, we iterate trhough the whole set
        else:
            loop_increment = int(len(df_filtered) / max_img_per_moa)+1 #otherwise we change the loop increment to only sample every ith picture and get a final number of pictures matching the limit
        print(moa)


        for n in range(0, len(df_filtered), loop_increment): #then we iterate through pictures per MoA
            
            path_dapi = df_filtered.Image_PathName_DAPI.iloc[n][6:] #and save the path of the respective image for all three channels
            dapi = df_filtered.Image_FileName_DAPI.iloc[n]
            path_dapi = '{}/{}/{}'.format(source_path, path_dapi, dapi)

            path_tubulin = df_filtered.Image_PathName_Tubulin.iloc[n][6:]
            tubulin = df_filtered.Image_FileName_Tubulin.iloc[n]
            path_tubulin = '{}/{}/{}'.format(source_path, path_tubulin, tubulin)
            
            path_actin = df_filtered.Image_PathName_Actin.iloc[n][6:]
            actin = df_filtered.Image_FileName_Actin.iloc[n]
            path_actin = '{}/{}/{}'.format(source_path, path_actin, actin)

            #then we copy each image into the proper folder following the structure showed above
            try:
                shutil.copy(path_dapi, '{}/{}/dapi/img{}.tif'.format(destination_path, moa, n))
            except IOError as io_err:
                os.makedirs(os.path.dirname('{}/{}/dapi/{}'.format(destination_path, moa, dapi))) #if the folder does not exist yet, we create it
                shutil.copy(path_dapi, '{}/{}/dapi/img{}.tif'.format(destination_path, moa, n))

            try:
                shutil.copy(path_tubulin, '{}/{}/tubulin/img{}.tif'.format(destination_path, moa, n))
            except IOError as io_err:
                os.makedirs(os.path.dirname('{}/{}/tubulin/{}'.format(destination_path, moa, tubulin)))
                shutil.copy(path_tubulin, '{}/{}/tubulin/img{}.tif'.format(destination_path, moa, n))
            
            try:
                shutil.copy(path_actin, '{}/{}/actin/img{}.tif'.format(destination_path, moa, n))
            except IOError as io_err:
                os.makedirs(os.path.dirname('{}/{}/actin/{}'.format(destination_path, moa, actin)))
                shutil.copy(path_actin, '{}/{}/actin/img{}.tif'.format(destination_path, moa, n))
        

def main():
    global project_config
    project_config = utl.load_project_conf(args.config)

    df_bbbc021 = pd.read_csv(args.datafile)

    min_img = df_bbbc021.groupby("Image_Metadata_Compound").count().TableNumber.min()
    max_img = df_bbbc021.groupby("Image_Metadata_Compound").count().TableNumber.max()

    test_cpd = df_bbbc021.groupby("Image_Metadata_MoA").Image_Metadata_Compound.apply(lambda x: x.sample(1)).reset_index(drop=True)
    test_cpd = test_cpd[test_cpd != "DMSO"]

    df_train = df_bbbc021[~df_bbbc021.Image_Metadata_Compound.isin(test_cpd)]
    df_test = df_bbbc021[df_bbbc021.Image_Metadata_Compound.isin(test_cpd)]
    
    #DMSO is special: it is the negative control. We want approx 20% of DMSO pics in the test set, and 80% in the train set
    df_test = pd.concat([df_test, df_bbbc021[df_bbbc021.Image_Metadata_Compound == "DMSO"][:300]])
    df_train = pd.concat([df_train, df_bbbc021[df_bbbc021.Image_Metadata_Compound == "DMSO"][300:]])

    sort_images(df_train, 
                args.image_dir, 
                os.path.join(args.image_dir, "sorted_reduced/train"), 
                max_img_per_moa=args.max_images)
    sort_images(df_test, 
                args.image_dir, 
                os.path.join(args.image_dir, "sorted_reduced/test"), 
                max_img_per_moa=args.max_images)


if __name__=="__main__":
    main()
