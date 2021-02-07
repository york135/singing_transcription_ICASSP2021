import argparse
import pickle
import json
from pathlib import Path
from data_utils import AudioDataset


def main(args):
    # Create dataset instances    
    print('Generating dataset...')
    print('Using directory: {}'.format(args.data_dir))   
    dataset = AudioDataset(args.data_dir, args.label_dir)

    # Write the datasets into binary files
    filename_trail = '_dataset.pkl'
    target_path = Path(args.output_dir) / (Path(args.data_dir).stem + filename_trail)
    with open(target_path, 'wb') as f:
        pickle.dump(dataset, f)
    print('Dataset generated at {}.'.format(target_path))


if __name__ == "__main__":
    """
    "data_dir": Directory to the dataset folder. It should contain several ".wav" files. 
    "label_dir": Directory to the dataset folder. It should contain several ".unv" files (groundtruth labels). 
    "output_dir": The dataset pkl file will be stored in "output_dir" folder.
    """

    parser = argparse.ArgumentParser(
        description="This script will read from a data directory and generate custom dataset class instance into a binary file.")
    parser.add_argument('data_dir')
    parser.add_argument('label_dir')
    parser.add_argument('output_dir')


    args = parser.parse_args()

    main(args)
