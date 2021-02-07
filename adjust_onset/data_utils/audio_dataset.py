from pathlib import Path
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import librosa
import os
import numpy as np
import random


def get_feature(y, sr):
    if sr != 16000:
        y = librosa.core.resample(y= y, orig_sr= sr, target_sr= 16000)

    y = librosa.util.normalize(y)

    # 20ms (320/16000) is the hop length of MIR-1k dataset labels.
    mfcc = librosa.feature.mfcc(y, sr=16000, n_fft=512, n_mfcc=20, hop_length=320, center=True, pad_mode='reflect')
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    feature = np.concatenate((mfcc, mfcc_delta, mfcc_delta2), axis=0)

    return feature


class AudioDataset(Dataset):
    """
    A TestDataset contains ALL frames for all songs, which is different from AudioDataset.
    """

    def __init__(self, data_dir, label_dir, is_test=False):
        self.data_instances = []

        for the_dir in tqdm(os.listdir(data_dir)):
            wav_path = os.path.join(data_dir, the_dir)
            y, sr = librosa.core.load(wav_path, sr=None, mono=True)
            features = get_feature(y, sr)

            # print (os.path.join(label_dir, the_dir[:-3]+ 'unv'))
            # print (wav_path)

            gt = np.loadtxt(os.path.join(label_dir, the_dir[:-3]+ 'unv'), dtype=int)

            # For each frame, combine adjacent frames as a data_instance
            feature_size, frame_num = features.shape[0], features.shape[1] - 1
            # print (feature_size, frame_num)
            for frame_idx in range(frame_num):
                concated_feature = torch.empty(feature_size, 11)
                for frame_window_idx in range(frame_idx - 5, frame_idx + 6):
                    # Boundary check
                    if frame_window_idx < 0:
                        choosed_idx = 0
                    elif frame_window_idx >= frame_num:
                        choosed_idx = frame_num - 1
                    else:
                        choosed_idx = frame_window_idx

                    concated_feature[:, frame_window_idx - frame_idx + 5] = torch.tensor(features[:, choosed_idx+1])

                if frame_idx >= gt.shape[0]:
                    self.data_instances.append([concated_feature, 1])
                else:
                    if gt[frame_idx] != 5:
                        self.data_instances.append([concated_feature, 0])
                    else:
                        self.data_instances.append([concated_feature, 1])


        print('Dataset initialized from {}.'.format(data_dir))

    def __getitem__(self, idx):
        return self.data_instances[idx]

    def __len__(self):
        return len(self.data_instances)


class TestDataset(Dataset):
    
    # no groundtruth is provided here.

    def __init__(self, data_dir, is_test=False):
        self.data_instances = []
        for song_dir in tqdm(sorted(Path(data_dir).iterdir())):  
            # wav_path = os.path.join(data_dir, song_dir)

            wav_path = song_dir / 'Vocal.wav'
            song_id = song_dir.stem
        
            y, sr = librosa.core.load(wav_path, sr=None, mono=True)
            features = get_feature(y, sr)

            # For each frame, combine adjacent frames as a data_instance
            feature_size, frame_num = features.shape[0], features.shape[1] - 1
            # print (feature_size, frame_num)
            for frame_idx in range(frame_num):
                concated_feature = torch.empty(feature_size, 11)
                for frame_window_idx in range(frame_idx - 5, frame_idx + 6):
                    # Boundary check
                    if frame_window_idx < 0:
                        choosed_idx = 0
                    elif frame_window_idx >= frame_num:
                        choosed_idx = frame_num - 1
                    else:
                        choosed_idx = frame_window_idx

                    concated_feature[:, frame_window_idx - frame_idx + 5] = torch.tensor(features[:, choosed_idx+1])

                self.data_instances.append([concated_feature, song_id])
                
        print('Dataset initialized from {}.'.format(data_dir))

    def __getitem__(self, idx):
        return self.data_instances[idx]

    def __len__(self):
        return len(self.data_instances)

