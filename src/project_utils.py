import tomlkit

import pandas as pd
import numpy as np

random_state = 42


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

    df_cmpds.index = pd.RangeIndex(len(df_cmpds.index))

    return df_cmpds


def create_train_test_val_sets(df):
    df_dataset_moa = df[df["Image_Metadata_MoA"].notna()]
    df_cmpd_moa = df_dataset_moa.groupby(["Image_Metadata_Compound", "Image_Metadata_MoA"])["Image_FileName_DAPI"].count().to_frame().reset_index()

    test_val_cmpds = list()
    for moa in df_cmpd_moa["Image_Metadata_MoA"].unique():
        if moa == "DMSO":
            continue
        test_val_cmpds.append(df_cmpd_moa[df_cmpd_moa["Image_Metadata_MoA"] == moa].sample(1, random_state=random_state)["Image_Metadata_Compound"].values[0])

    df_train = df_dataset_moa.drop(df_dataset_moa[df_dataset_moa["Image_Metadata_Compound"].isin(test_val_cmpds)].index)

    # balance dataset, sample DMSO and taxol
    # handle DMSO
    df_dmso = df_train[df_train["Image_Metadata_Compound"] == "DMSO"].sample(n=200, random_state=random_state)
    df_train = df_train[df_train["Image_Metadata_Compound"] != "DMSO"]
    df_train = pd.concat([df_train, df_dmso], axis=0)

    # handel taxol
    df_taxol = df_train[df_train["Image_Metadata_Compound"] == "taxol"].sample(n=100, random_state=random_state)
    df_train = df_train[df_train["Image_Metadata_Compound"] != "taxol"]
    df_train = pd.concat([df_train, df_taxol], axis=0, ignore_index=True) # .reset_index()

    df_val_test = df_dataset_moa.drop(df_dataset_moa[~df_dataset_moa["Image_Metadata_Compound"].isin(test_val_cmpds)].index)

    df_cmpd_moa_tst_val = df_cmpd_moa.loc[df_cmpd_moa["Image_Metadata_Compound"].isin(test_val_cmpds)]

    df_val = pd.DataFrame()
    for cmpd in test_val_cmpds:
        num_samples = int(df_cmpd_moa_tst_val[df_cmpd_moa_tst_val["Image_Metadata_Compound"] == cmpd]["Image_FileName_DAPI"].values[0] / 2)
        df_val = pd.concat([df_val, df_val_test[df_val_test["Image_Metadata_Compound"] == cmpd].sample(num_samples, random_state=random_state)], axis=0)

    df_test = df_val_test.drop(df_val.index)

    # add DMSO images to test and val
    dmso_train_list = df_dmso["Image_FileName_DAPI"].to_list()
    df_dmso_full = df_dataset_moa[df_dataset_moa["Image_Metadata_Compound"] == "DMSO"]
    df_dmso_red = df_dmso_full.drop(df_dmso_full[df_dmso_full["Image_FileName_DAPI"].isin(dmso_train_list)].index)
    df_dmso_red = df_dmso_red.sample(n=96, random_state=random_state)

    df_test = pd.concat([df_test, df_dmso_red.iloc[:48,:]], axis=0, ignore_index=True)
    df_val = pd.concat([df_val, df_dmso_red.iloc[48:,:]], axis=0, ignore_index=True)

    return df_train, df_val, df_test
