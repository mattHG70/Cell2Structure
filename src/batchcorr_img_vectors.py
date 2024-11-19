import re
import argparse

import pandas as pd
from inmoose.pycombat import pycombat_norm


parser = argparse.ArgumentParser(description="Batch correction of the embedding vector fle using PyCombat")
parser.add_argument('-infile', type=str, required=True, help="Input file name, full path")
parser.add_argument('-outfile', type=str, required=True, help="Output file name, full path")
args = parser.parse_args()


def main():
    # read in embedding file. Set smiles and moa column to datatype string
    # because Pandas finds mixed datatypes due to nan values.
    df_embed_vec = pd.read_csv(args.infile)

    # split in data and batches (plates)
    vec_cols = [c for c in df_embed_vec.columns if re.match(r"[V]\d+", c)]
    data = df_embed_vec[vec_cols]
    batches = df_embed_vec.loc[:, "Image_Metadata_Plate_DAPI"]
    
    # actual batch correction step
    data_corrected = pycombat_norm(data.T, batches)

    # concatenate corrected vectors with metadata
    df_embed_vec_corr = pd.concat([df_embed_vec.iloc[:,:15], data_corrected.T], axis=1)

    # write out to CSV file
    df_embed_vec_corr.to_csv(args.outfile, index=False)
 
    
if __name__=="__main__":
    main()