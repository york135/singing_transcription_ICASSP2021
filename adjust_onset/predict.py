import argparse
import json
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # nopep8
from data_utils.audio_dataset import TestDataset
from predictor import UnvoicedPredictor
from pathlib import Path
import pickle
import time

from tqdm import tqdm
import numpy as np

def main(args):
    # Create predictor
    # print (time.time())
    device= 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
    # print ("use", device)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    predictor = UnvoicedPredictor(device=device, model_path=args.model_path)
    # print('Creating testing dataset...')

    filename_trail = '_test_dataset_new.pkl'
    target_path = Path("./") / (Path(args.test_dir).stem + filename_trail)
    
    if os.path.isfile(target_path):
        # read from pickle
        with open(target_path, 'rb') as f:
            test_dataset = pickle.load(f)

    else:
        # Read from test_dir
        test_dataset = TestDataset(args.test_dir)

        with open(target_path, 'wb') as f:
            pickle.dump(test_dataset, f, protocol=4)

    print('Dataset generated at {}.'.format(target_path))

    """
    # search for the best prior probability (that makes precision equals to recall)
    candidate = [(i-80.0)/20.0 for i in range(0, 101)]
    confusion_list = []
    for prior_prob in candidate:
        results, sequences = predictor.predict_test(test_dataset, prior_prob=prior_prob)

        confusion = np.zeros((2, 2))
        for song_id in sequences.keys():
            # get the groundtruth
            real = np.loadtxt(os.path.join("val_label", song_id[:-3]+ 'unv'), dtype=int)

            for i in range(len(real)):
                if real[i] == 5:
                    if sequences[song_id][i] == 1:
                        confusion[0][0] = confusion[0][0] + 1
                    else:
                        confusion[0][1] = confusion[0][1] + 1

                else:
                    if sequences[song_id][i] == 1:
                        confusion[1][0] = confusion[1][0] + 1
                    else:
                        confusion[1][1] = confusion[1][1] + 1

        print ("Prior_prob:", prior_prob)
        print (confusion)
        confusion_list.append(confusion)
    """


    prior_prob = -1.16
    results, sequences = predictor.predict_test(test_dataset, prior_prob=prior_prob)

    # Write results to target file
    
    with open(args.predict_file, 'w') as f:
        output_string = json.dumps(results)
        f.write(output_string)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dir')
    parser.add_argument('predict_file')
    parser.add_argument('model_path')
    
    args = parser.parse_args()

    main(args)
