import os
import requests
import argparse

import project_utils as utl

from tqdm import tqdm


# default parameters
default_data_dir = "../data/external"
default_config_file = "../project_config.toml"


parser = argparse.ArgumentParser(prog="get_dataset.py", 
                                 description="Download BBBC021_v1 CSV files to data directory")
parser.add_argument('-datadir', 
                    type=str, 
                    required=False, 
                    default=default_data_dir,
                    help="Data directory in which the files should be stored")
parser.add_argument('-config', 
                    type=str, 
                    required=False, 
                    default=default_config_file,
                    help="Project configuration file (toml format)")
args = parser.parse_args()


# Download the respective csv file and place it into the
# external data folder
def get_BBBC021_dataset(url, filepath):
    resp = requests.get(url, stream=True)
    with open(filepath, "wb") as file_out:
        for data in tqdm(resp.iter_content()):
            file_out.write(data)
    

def main():
    data_dir = args.datadir
    config_file = args.config

    # Load the project configuraion containing the csv file names
    config = utl.load_project_conf(config_file)
    
    # Go through all configured csv files and download them from
    # the Broad Institute
    for data_table in config["BBC021_CSV"]:
        if data_table["type"] == "moas_enchanced":
            continue
        out_filepath = os.path.join(data_dir, data_table["file"])
        print(data_table["file"])
        get_BBBC021_dataset(data_table["url"], out_filepath)


if __name__=="__main__":
    main()