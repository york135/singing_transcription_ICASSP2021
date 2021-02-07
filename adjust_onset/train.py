import torch
import torch.nn as nn
import argparse
from predictor import UnvoicedPredictor
import os

def main(args):

    device= 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.set_device(0)
    print ("use", device)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    predictor = UnvoicedPredictor(device=device)
    predictor.fit(
        train_dataset_path=args.training_dataset,
        valid_dataset_path=args.validation_dataset,
        model_dir=args.model_dir,
        batch_size=100,
        valid_batch_size=100,
        epoch=10,
        lr=1e-4,
        save_every_epoch=1,
    )


if __name__ == '__main__':
    """
    This script performs training and validation of the unvoiced frame detector.
    training_dataset: The pkl file used as training data.
    validation_dataset: The pkl file used as validation data.
    model_dir: The directory that stores models.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('training_dataset')
    parser.add_argument('validation_dataset')
    parser.add_argument('model_dir')

    args = parser.parse_args()

    main(args)
