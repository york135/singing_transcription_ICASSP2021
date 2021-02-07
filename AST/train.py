import torch
import torch.nn as nn
import argparse
from predictor import EffNetPredictor
import os

def main(args):

    device= 'cpu'
    if torch.cuda.is_available():
        device = args.device
    print ("use", device)

    predictor = EffNetPredictor(device=device, model_path=args.model_path)
    predictor.fit(
        train_dataset_path=args.training_dataset,
        valid_dataset_path=args.validation_dataset,
        model_dir=args.model_dir,
        batch_size=50,
        valid_batch_size=200,
        epoch=10,
        lr=1e-4,
        save_every_epoch=1,
        save_prefix=args.save_prefix
    )


if __name__ == '__main__':
    """
    This script performs training and validation of the singing transcription model.
    training_dataset: The pkl file used as training data.
    validation_dataset: The pkl file used as validation data.
    model_dir: The directory that stores models.
    save_prefix: The prefix of the models.
    device: The device (e.g. cuda:0) to use if cuda is available.
    model-path: Pre-trained model (optional). If provided, the weights of the pre-trained model will be loaded, and used as the initial weights.

    Sample usage:
    python train.py train.pkl test.pkl models/ effnet cuda:0
    The script will use train.pkl as training set, test.pkl as validation set (it will predict the whole validation set every epoch, and print the validation loss).
    Each epoch, a model file called called "effnet_{epoch}" will be saved at "models/".
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('training_dataset')
    parser.add_argument('validation_dataset')
    parser.add_argument('model_dir')
    parser.add_argument('save_prefix')
    parser.add_argument('device')
    parser.add_argument('--model-path')

    args = parser.parse_args()

    main(args)
