import argparse
import numpy as np
import pandas as pd


# default parameters
default_embd_path = "../data/interim/embeddings/original_model.npy"
default_dataset = "../data/processed/BBBC021_final_original_dataset.csv"
default_output_embd = "../data/embeddings/embeddings_original_MoA_only"


parser = argparse.ArgumentParser(prog="merge_embeddings.py", 
                                 description="Merge embedding vectors with dataset")
parser.add_argument('-intm_embd', 
                    type=str, 
                    required=False, 
                    default=default_embd_path,
                    help="Interim embedding vectors")
parser.add_argument('-dataset', 
                    type=str, 
                    required=False, 
                    default=default_dataset,
                    help="Dataset containing image and metadata")
parser.add_argument('-out_embd', 
                    type=str, 
                    required=False, 
                    default=default_output_embd,
                    help="Output embedding file without file type")
args = parser.parse_args()


"""
Merge the image embedding vectors with the BBBC021 dataset.
Image embedding vectors are stored in a Numpy array.
"""
def main():
    df = pd.read_csv(args.dataset)
    df = df[df["Image_Metadata_MoA"].notna()].reset_index(drop=True)

    vectors = np.load(args.intm_embd)
    vector_cols = ["V"+str(c) for c in range(vectors.shape[1])]
    df_vectors = pd.DataFrame(data=vectors, columns=vector_cols)

    df_merged = pd.concat([df, df_vectors], axis=1)

    # export embedding dataset as csv and compressed parquet file
    df_merged.to_csv(args.out_embd + ".csv")
    df_merged.to_parquet(args.out_embd + ".parquet")


if __name__=="__main__":
    main()