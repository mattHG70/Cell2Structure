import os
import tomlkit
import argparse

import pandas as pd


default_config_file = "../project_config.toml"
default_data_ext_dir = "../data/external"
default_data_interim_dir = "../data/interim"
default_data_final_dir = "../data/processed"


parser = argparse.ArgumentParser(prog="preproc_dataset.py", 
                                 description="Prepocess and prepare BBBC021_v1 dataset")
parser.add_argument('-datadir_ext', 
                    type=str, 
                    required=False, 
                    default=default_data_ext_dir,
                    help="Data directory containing the original BBBC021 dataset")
parser.add_argument('-datadir_intm', 
                    type=str, 
                    required=False, 
                    default=default_data_interim_dir,
                    help="Data directory storing intermediate files")
parser.add_argument('-datadir_final', 
                    type=str, 
                    required=False, 
                    default=default_data_final_dir,
                    help="Data directory in which the processed files are stored")
parser.add_argument('-config', 
                    type=str, 
                    required=False, 
                    default=default_config_file,
                    help="Project configuration file (toml format)")
args = parser.parse_args()


def load_project_conf(toml_file):
    with open(toml_file, "rb") as config_file:
        config = tomlkit.load(config_file)
    return config


def get_config_item(type):
    for item in project_config["BBC021_CSV"]:
        if item["type"] == type:
            return item


def load_dataframe(data_dir, type):
    path = os.path.join(data_dir, get_config_item(type)["file"])
    df = pd.read_csv(path)
    return df


def prepare_original_dataset(df):
    df_compounds = load_dataframe(args.datadir_ext, "compounds")
    df_moas = load_dataframe(args.datadir_ext, "moas")
    
    # add DMSO SMILES to compounds
    df_compounds.loc[df_compounds["compound"] == "DMSO", "smiles"] = project_config["structures"]["DMSO"]

    # remove compounds missing a SMILES
    df_compounds = df_compounds[df_compounds["smiles"].notna()]

    # remove concentraion column from MoAs (included in images too) and remove duplicates
    df_moas = df_moas.drop(columns=["concentration"]).drop_duplicates(ignore_index=True)

    # merge MoAs dataframe
    df = pd.merge(df, df_moas, how="left", left_on="Image_Metadata_Compound", right_on="compound")
    df = df.drop(columns=["compound"])

    # merge compounds dataframe
    df = pd.merge(df, df_compounds, how="left", left_on="Image_Metadata_Compound", right_on="compound")
    df = df.drop(columns=["compound"])

    # rename columns to meet the original naming scheme
    df = df.rename(columns={"moa": "Image_Metadata_MoA", "smiles": "Image_Metadata_SMILES"})

    output_path = os.path.join(args.datadir_final, project_config["final_dataset"]["file_original"])
    df.to_csv(output_path, index=False)


def prepare_enhanced_dataseet(df):
    df_compounds = load_dataframe(args.datadir_ext, "compounds")

    # add DMSO SMILES to compounds
    df_compounds.loc[df_compounds["compound"] == "DMSO", "smiles"] = project_config["structures"]["DMSO"]

    # remove compounds missing a SMILES
    df_compounds = df_compounds[df_compounds["smiles"].notna()]

    # load enhanced MoAs dataset (Excel file)
    enhanced_ds_path = os.path.join(args.datadir_intm, get_config_item("moas_enhanced")["file"])
    df_moas_enhanced = pd.read_excel(enhanced_ds_path)
    df_moas_enhanced = df_moas_enhanced.drop(columns=["smiles"])

    # merge enhanced MoAs dataframe
    df = pd.merge(df, df_moas_enhanced, how="left", left_on="Image_Metadata_Compound", right_on="compound")
    df = df.drop(columns=["compound"])
    
    # merge compounds dataframe
    df = pd.merge(df, df_compounds, how="left", left_on="Image_Metadata_Compound", right_on="compound")
    df = df.drop(columns=["compound"])

    # rename columns to meet the original naming scheme
    df = df.rename(columns={"moa": "Image_Metadata_MoA", "smiles": "Image_Metadata_SMILES"})

    output_path = os.path.join(args.datadir_final, project_config["final_dataset"]["file_enhanced"])
    df.to_csv(output_path, index=False)
    

def main():
    global project_config
    project_config = load_project_conf(args.config)
    
    df_bbbc021 = load_dataframe(args.datadir_ext, "images")

    prepare_original_dataset(df_bbbc021.copy())

    prepare_enhanced_dataseet(df_bbbc021.copy())


if __name__=="__main__":
    main()