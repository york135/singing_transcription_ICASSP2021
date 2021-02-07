import argparse
import pickle
import json
from pathlib import Path
from data_utils import AudioDataset
import joblib

import warnings
warnings.filterwarnings('ignore')

def main(args):
    # Create dataset instances    
    print('Generating dataset...')
    print('Using directory: {}'.format(args.data_dir))

    # Write the datasets into binary files
    filename_trail = args.filename_trail
    target_path = Path(args.output_dir) / (Path(args.data_dir).stem + filename_trail)

    dataset = AudioDataset(gt_path=args.gt_path, data_dir=args.data_dir)

    with open(target_path, 'wb') as f:
        pickle.dump(dataset, f)
    print('Dataset generated at {}.'.format(target_path))


if __name__ == "__main__":
    """
    "data_dir": Directory to the dataset folder. It should contain several subdirectories. Each subdirectory should contains an audio file called "Vocal.wav". 
    "gt_path": Path to the groundtruth JSON file.
    "output_dir" and "filename_trail": The dataset pkl file will be stored in "output_dir" folder, and the file name will be "data_dir.stem + filename_trail"
    e.g. python generate_dataset.py dataset/train/ MIR-ST500_corrected_1005.json ./ _dataset.pkl
    This will generate "train_dataset.pkl" to "./"
    """
    parser = argparse.ArgumentParser(
        description="This script will read from a data directory and generate custom dataset class instance into a binary file.")
    parser.add_argument('data_dir')
    parser.add_argument('gt_path')
    parser.add_argument('output_dir')
    parser.add_argument('filename_trail')

    args = parser.parse_args()

    main(args)
