from pathlib import Path
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import librosa
import os
import numpy as np
import random
import concurrent.futures
import pickle
import time
import json

def preprocess(gt_data, length, pitch_shift=0):

    new_label = []

    cur_note = 0
    cur_note_onset = gt_data[cur_note][0]
    cur_note_offset = gt_data[cur_note][1]
    cur_note_pitch = gt_data[cur_note][2] + pitch_shift

    # start from C2 (36) to B5 (83), total: 4 classes. This is a little confusing
    octave_start = 0
    octave_end = 3
    pitch_class_num = 12
    frame_size = 1024.0 / 44100.0

    for i in range(length):
        cur_time = i * frame_size

        if abs(cur_time - cur_note_onset) <= (frame_size / 2.0):
            # First dim : onset
            # Second dim : no pitch
            if i == 0 or new_label[-1][0] != 1:
                my_oct = int(min(max(octave_start, (cur_note_pitch- 36)//pitch_class_num), octave_end)) - octave_start
                my_pitch_class = cur_note_pitch % pitch_class_num
                label = [1, 0, my_oct, my_pitch_class]
                new_label.append(label)
            else:
                my_oct = int(min(max(octave_start, (cur_note_pitch- 36)//pitch_class_num), octave_end)) - octave_start
                my_pitch_class = cur_note_pitch % pitch_class_num
                label = [0, 0, my_oct, my_pitch_class]
                new_label.append(label)

        elif cur_time < cur_note_onset or cur_note >= len(gt_data):
            # For the frame that doesn't belong to any note
            label = [0, 1, octave_end+1, pitch_class_num]
            new_label.append(label)

        elif abs(cur_time - cur_note_offset) <= (frame_size / 2.0):
            # For the offset frame
            my_oct = int(min(max(octave_start, (cur_note_pitch- 36)//pitch_class_num), octave_end)) - octave_start
            my_pitch_class = cur_note_pitch % pitch_class_num
            label = [0, 1, my_oct, my_pitch_class]

            cur_note = cur_note + 1
            if cur_note < len(gt_data):
                cur_note_onset = gt_data[cur_note][0]
                cur_note_offset = gt_data[cur_note][1]
                cur_note_pitch = gt_data[cur_note][2] + pitch_shift
                if abs(cur_time - cur_note_onset)  <= (frame_size / 2.0):
                    my_oct = int(min(max(octave_start, (cur_note_pitch- 36)//pitch_class_num), octave_end)) - octave_start
                    my_pitch_class = cur_note_pitch % pitch_class_num
                    label[0] = 1
                    label[1] = 0
                    label[2] = my_oct
                    label[3] = my_pitch_class

            new_label.append(label)

        else:
            # For the voiced frame
            my_oct = int(min(max(octave_start, (cur_note_pitch- 36)//pitch_class_num), octave_end)) - octave_start
            my_pitch_class = cur_note_pitch % pitch_class_num

            label = [0, 0, my_oct, my_pitch_class]
            new_label.append(label)

    return np.array(new_label)


def get_cqt(y, filter_scale=1):
    return np.abs(librosa.cqt(y, sr=44100, hop_length=1024, fmin=librosa.midi_to_hz(36)
        , n_bins=84*2, bins_per_octave=12*2, filter_scale=filter_scale)).T


def get_feature(y):
    y = librosa.util.normalize(y)
    cqt_feature = get_cqt(y, 1.0)
    return torch.tensor(cqt_feature, dtype=torch.float).unsqueeze(1)

class AudioDataset(Dataset):

    def __init__(self, gt_path, data_dir=None):

        with open(gt_path) as json_data:
            gt = json.load(json_data)
        
        self.data_instances = []
        self.answer_instances = []
        self.pitch_instances = []
        total_count = len(os.listdir(data_dir))
        
        temp_cqt = {}
        future = {}
        print ("computing CQT......")

        frame_size = 1024.0 / 44100.0

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for the_dir in os.listdir(data_dir):
                wav_path = os.path.join(data_dir, the_dir, "Vocal.wav")

                y, sr = librosa.core.load(wav_path, sr=None, mono=True)
                if sr != 44100:
                    y = librosa.core.resample(y= y, orig_sr= sr, target_sr= 44100)
                future[the_dir] = executor.submit(get_feature, y)

        for the_dir in os.listdir(data_dir):
            temp_cqt[the_dir] = future[the_dir].result()

        print ("creating dataset......")

        for the_dir in tqdm(os.listdir(data_dir)):
            cqt_data = temp_cqt[the_dir]

            gt_data = gt[the_dir]
            answer_data = preprocess(gt_data, cqt_data.shape[0])

            frame_num, channel_num, cqt_size = cqt_data.shape[0], cqt_data.shape[1], cqt_data.shape[2]

            my_padding = torch.zeros((cqt_data.shape[1], cqt_data.shape[2]), dtype=torch.float)

            for frame_idx in range(frame_num):
                cqt_feature = []
                for frame_window_idx in range(frame_idx-5, frame_idx+6):
                    # padding with zeros if needed
                    if frame_window_idx < 0 or frame_window_idx >= frame_num:
                        cqt_feature.append(my_padding.unsqueeze(1))
                    else:
                        choosed_idx = frame_window_idx
                        cqt_feature.append(cqt_data[choosed_idx].unsqueeze(1))

                cqt_feature = torch.cat(cqt_feature, dim=1)
                self.data_instances.append((cqt_feature, answer_data[frame_idx]))


    def __getitem__(self, idx):
        return self.data_instances[idx]

    def __len__(self):
        return len(self.data_instances)