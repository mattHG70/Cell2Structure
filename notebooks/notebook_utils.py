import pandas as pd
import numpy as np

random_state = 42


def create_train_test_val_sets(df):
    df_dataset_moa = df[df["Image_Metadata_MoA"].notna()]
    df_cmpd_moa = df_dataset_moa.groupby(["Image_Metadata_Compound", "Image_Metadata_MoA"])["Image_FileName_DAPI"].count().to_frame().reset_index()

    test_val_cmpds = list()
    for moa in df_cmpd_moa["Image_Metadata_MoA"].unique():
        if moa == "DMSO":
            continue
        test_val_cmpds.append(df_cmpd_moa[df_cmpd_moa["Image_Metadata_MoA"] == moa].sample(1, random_state=random_state)["Image_Metadata_Compound"].values[0])

    df_train = df_dataset_moa.drop(df_dataset_moa[df_dataset_moa["Image_Metadata_Compound"].isin(test_val_cmpds)].index)
    df_val_test = df_dataset_moa.drop(df_dataset_moa[~df_dataset_moa["Image_Metadata_Compound"].isin(test_val_cmpds)].index)

    df_cmpd_moa_tst_val = df_cmpd_moa.loc[df_cmpd_moa["Image_Metadata_Compound"].isin(test_val_cmpds)]

    df_val = pd.DataFrame()
    for cmpd in test_val_cmpds:
        num_samples = int(df_cmpd_moa_tst_val[df_cmpd_moa_tst_val["Image_Metadata_Compound"] == cmpd]["Image_FileName_DAPI"].values[0] / 2)
        df_val = pd.concat([df_val, df_val_test[df_val_test["Image_Metadata_Compound"] == cmpd].sample(num_samples, random_state=random_state)], axis=0)

    df_test = df_val_test.drop(df_val.index)

    return df_train, df_val, df_test
