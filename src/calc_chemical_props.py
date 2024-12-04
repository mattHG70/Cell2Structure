import argparse

import pandas as pd
import numpy as np
import project_utils as utl

from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools


default_data_file = "../data/processed/BBBC021_final_original_dataset.csv"
default_config_file = "../project_config.toml"
default_output_file = "../data/processed/BBBC021_original_compounds_props.pickle"
default_style_option = "moa"


parser = argparse.ArgumentParser(prog="calc_chemical_props.py", 
                                 description="Calculate RDKit properties for BBBC021_v1 compounds")
parser.add_argument('-style', 
                    type=str, 
                    choices=["moa", "all"],
                    required=False, 
                    default=default_style_option,
                    help="Take all SMILES or only ones mapped to MoA")
parser.add_argument('-datafile', 
                    type=str, 
                    required=False, 
                    default=default_data_file,
                    help="Data file containing compounds and SMILES")
parser.add_argument('-outputfile', 
                    type=str, 
                    required=False, 
                    default=default_output_file,
                    help="File to output compounds and their RDKit properties")
parser.add_argument('-config', 
                    type=str, 
                    required=False, 
                    default=default_config_file,
                    help="Project configuration file (toml format)")
args = parser.parse_args()


# calculat RDKit descriptors and add them to Pandas dataframe
def generate_rdkit_descr(df):
    mols = df["mol"].to_list()
    descriptors = [Descriptors.CalcMolDescriptors(mol) for mol in mols]
    df = pd.DataFrame(data=descriptors)

    return df

        
def main():
    global project_config
    project_config = utl.load_project_conf(args.config)

    df_bbbc021 = pd.read_csv(args.datafile)

    df_compounds = utl.get_dataframe(df_bbbc021, args.style)

    # add RDKit molecule column to dataframe
    PandasTools.AddMoleculeColumnToFrame(df_compounds, smilesCol="Image_Metadata_SMILES", molCol="mol")

    descriptors = generate_rdkit_descr(df_compounds)

    df_compounds = pd.concat([df_compounds, descriptors], axis=1)

    df_compounds.to_pickle(args.outputfile)


if __name__=="__main__":
    main()