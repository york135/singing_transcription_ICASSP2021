from pathlib import Path
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import librosa
import os
import numpy as np
import random
from .audio_dataset import get_feature

def do_svs_spleeter(y, sr):
    from spleeter.separator import Separator
    import warnings
    separator = Separator('spleeter:2stems')
    warnings.filterwarnings('ignore')

    if sr != 44100:
        y = librosa.core.resample(y=y, orig_sr=sr, target_sr=44100)

    waveform = np.expand_dims(y, axis=1)

    prediction = separator.separate(waveform)
    ret = librosa.core.to_mono(prediction["vocals"].T)
    ret = np.clip(ret, -1.0, 1.0)
    del separator
    return ret, 44100


class SeqDataset(Dataset):

    def __init__(self, wav_path, song_id, is_test=False, do_svs=False):

        y, sr = librosa.core.load(wav_path, sr=None, mono=True)
        if sr != 44100:
            y = librosa.core.resample(y= y, orig_sr= sr, target_sr= 44100)
        y = librosa.util.normalize(y)

        if do_svs == True:
            y, sr = do_svs_spleeter(y, sr)

        self.data_instances = []

        cqt_data = get_feature(y)

        frame_size = 1024.0 / 44100.0

        frame_num, channel_num, cqt_size = cqt_data.shape[0], cqt_data.shape[1], cqt_data.shape[2]

        my_padding = torch.zeros((cqt_data.shape[1], cqt_data.shape[2]), dtype=torch.float)

        for frame_idx in range(frame_num):
            cqt_feature = []
            for frame_window_idx in range(frame_idx - 5, frame_idx + 6):
                # padding with zeros if needed
                if frame_window_idx < 0 or frame_window_idx >= frame_num:
                    cqt_feature.append(my_padding.unsqueeze(1))
                else:
                    choosed_idx = frame_window_idx
                    cqt_feature.append(cqt_data[choosed_idx].unsqueeze(1))

            cqt_feature = torch.cat(cqt_feature, dim=1)
            self.data_instances.append((cqt_feature, song_id))


    def __getitem__(self, idx):
        return self.data_instances[idx]

    def __len__(self):
        return len(self.data_instances)
