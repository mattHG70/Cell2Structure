import os
import requests
import argparse

from tqdm import tqdm


default_data_dir = "../data/external"
dataset_urls = {
    "compounds": {"url": "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_compound.csv", "file": "BBBC021_v1_compound.csv"},
    "images": {"url": "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_image.csv", "file": "BBBC021_v1_image.csv"},
    "moas": {"url": "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_moa.csv", "file": "BBBC021_v1_moa.csv"}
}


parser = argparse.ArgumentParser(prog="get_dataset.py", 
                                 description="Download BBBC021_v1 CSV files to data directory")
parser.add_argument('-datadir', 
                    type=str, 
                    required=False, 
                    default=default_data_dir,
                    help="Data directory in which the files should be stored")
args = parser.parse_args()


def get_BBBC021_dataset(url, filepath):
    resp = requests.get(url, stream=True)
    with open(filepath, "wb") as file_out:
        for data in tqdm(resp.iter_content()):
            file_out.write(data)


def main():
    data_dir = args.datadir
    for key, value in dataset_urls.items():
        out_filepath = os.path.join(data_dir, value["file"])
        print(value["file"])
        get_BBBC021_dataset(value["url"], out_filepath)


if __name__=="__main__":
    main()