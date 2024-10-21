import tomlkit

import pandas as pd


def load_project_conf(toml_file):
    with open(toml_file, "rb") as config_file:
        config = tomlkit.load(config_file)
    
    return config


def get_dataframe(df, style):
    df_cmpds = df[["Image_Metadata_Compound", "Image_Metadata_MoA", "Image_Metadata_SMILES"]].copy()
    df_cmpds = df_cmpds.drop_duplicates(ignore_index=True)
    df_cmpds = df_cmpds[df_cmpds["Image_Metadata_SMILES"].notna()]
    
    if style == "moa":
        df_cmpds = df_cmpds[(df_cmpds["Image_Metadata_SMILES"].notna()) & (df_cmpds["Image_Metadata_MoA"].notna())]

    return df_cmpds