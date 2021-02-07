import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import librosa
import time
from pathlib import Path
import pickle
from tqdm import tqdm
from collections import Counter
import numpy as np

import sys
import os

from net import EffNetb0
import math
from data_utils import AudioDataset

FRAME_LENGTH = librosa.frames_to_time(1, sr=44100, hop_length=1024)


class EffNetPredictor:
    def __init__(self, device= "cuda:0", model_path=None):
        """
        Params:
        model_path: Optional pretrained model file
        """
        # Initialize model
        self.device = device

        if model_path is not None:
            self.model = EffNetb0().to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
            print('Model read from {}.'.format(model_path))

        else:            
            self.model = EffNetb0().to(self.device)

        print('Predictor initialized.')


    def fit(self, train_dataset_path, valid_dataset_path, model_dir, **training_args):
        """
        train_dataset_path: The path to the training dataset.pkl
        valid_dataset_path: The path to the validation dataset.pkl
        model_dir: The directory to save models for each epoch
        training_args:
          - batch_size
          - valid_batch_size
          - epoch
          - lr
          - save_every_epoch
        """
        # Set paths
        self.train_dataset_path = train_dataset_path
        self.valid_dataset_path = valid_dataset_path
        self.model_dir = model_dir
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        # Set training params
        self.batch_size = training_args['batch_size']
        self.valid_batch_size = training_args['valid_batch_size']
        self.epoch = training_args['epoch']
        self.lr = training_args['lr']
        self.save_every_epoch = training_args['save_every_epoch']

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.onset_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0,], device=self.device))
        self.offset_criterion = nn.BCEWithLogitsLoss()

        self.octave_criterion = nn.CrossEntropyLoss(ignore_index=100)
        self.pitch_criterion = nn.CrossEntropyLoss(ignore_index=100)

        # Read the datasets
        print('Reading datasets...')
        print ('cur time: %.6f' %(time.time()))


        with open(self.train_dataset_path, 'rb') as f:
            self.training_dataset = pickle.load(f)

        with open(self.valid_dataset_path, 'rb') as f:
            self.validation_dataset = pickle.load(f)

        self.train_loader = DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

        self.valid_loader = DataLoader(
            self.validation_dataset,
            batch_size=self.valid_batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

        start_time = time.time()
        # Start training
        print('Start training...')
        print ('cur time: %.6f' %(time.time()))
        self.iters_per_epoch = len(self.train_loader)
        print (self.iters_per_epoch)

        for epoch in range(1, self.epoch + 1):
            self.model.train()

            total_training_loss = 0
            total_split_loss = np.zeros(4)            

            for batch_idx, batch in enumerate(self.train_loader):
                # Parse batch data
                
                input_tensor = batch[0].to(self.device)
                onset_prob = batch[1][:, 0].float().to(self.device)
                offset_prob = batch[1][:, 1].float().to(self.device)
                pitch_octave = batch[1][:, 2].long().to(self.device)
                pitch_class = batch[1][:, 3].long().to(self.device)

                loss = 0                
                self.optimizer.zero_grad()

                onset_logits, offset_logits, pitch_octave_logits, pitch_class_logits = self.model(input_tensor)

                split_train_loss0 = self.onset_criterion(onset_logits, onset_prob)
                split_train_loss1 = self.offset_criterion(offset_logits, offset_prob)
                split_train_loss2 = self.octave_criterion(pitch_octave_logits, pitch_octave)
                split_train_loss3 = self.pitch_criterion(pitch_class_logits, pitch_class)

               
                
                total_split_loss[0] = total_split_loss[0] + split_train_loss0.item() 
                total_split_loss[1] = total_split_loss[1] + split_train_loss1.item()
                total_split_loss[2] = total_split_loss[2] + split_train_loss2.item()
                total_split_loss[3] = total_split_loss[3] + split_train_loss3.item()

                loss = split_train_loss0 + split_train_loss1 + split_train_loss2 + split_train_loss3
                loss.backward()
                self.optimizer.step()
                total_training_loss += loss.item()
                
                if batch_idx % 5000 == 0 and batch_idx != 0:
                    print (epoch, batch_idx, "time:", time.time()-start_time, "loss:", total_training_loss / (batch_idx+1))


            if epoch % self.save_every_epoch == 0:
                # Perform validation
                self.model.eval()
                with torch.no_grad():
                    total_valid_loss = 0
                    split_val_loss = np.zeros(6)
                    for batch_idx, batch in enumerate(self.valid_loader):

                        input_tensor = batch[0].to(self.device)

                        onset_prob = batch[1][:, 0].float().to(self.device)
                        offset_prob = batch[1][:, 1].float().to(self.device)
                        pitch_octave = batch[1][:, 2].long().to(self.device)
                        pitch_class = batch[1][:, 3].long().to(self.device)

                        onset_logits, offset_logits, pitch_octave_logits, pitch_class_logits = self.model(input_tensor)

                        split_val_loss0 = self.onset_criterion(onset_logits, onset_prob)
                        split_val_loss1 = self.offset_criterion(offset_logits, offset_prob)

                        split_val_loss2 = self.octave_criterion(pitch_octave_logits, pitch_octave)
                        split_val_loss3 = self.pitch_criterion(pitch_class_logits, pitch_class)

                        split_val_loss[0] = split_val_loss[0] + split_val_loss0.item()
                        split_val_loss[1] = split_val_loss[1] + split_val_loss1.item()
                        split_val_loss[2] = split_val_loss[2] + split_val_loss2.item()  
                        split_val_loss[3] = split_val_loss[3] + split_val_loss3.item()

                        
                        # Calculate loss
                        loss = split_val_loss0 + split_val_loss1 + split_val_loss2 + split_val_loss3
                        total_valid_loss += loss.item()


                # Save model
                save_dict = self.model.state_dict()
                target_model_path = Path(self.model_dir) / (training_args['save_prefix']+'_{}'.format(epoch))
                torch.save(save_dict, target_model_path)

                # Epoch statistics
                print(
                    '| Epoch [{:4d}/{:4d}] Train Loss {:.4f} Valid Loss {:.4f} Time {:.1f}'.format(
                        epoch,
                        self.epoch,
                        total_training_loss / len(self.train_loader),
                        total_valid_loss / len(self.valid_loader),
                        time.time()-start_time))

                print('split train loss: onset {:.4f} offset {:.4f} pitch octave {:.4f} pitch class {:.4f}'.format(
                        total_split_loss[0]/len(self.train_loader),
                        total_split_loss[1]/len(self.train_loader),
                        total_split_loss[2]/len(self.train_loader),
                        total_split_loss[3]/len(self.train_loader)
                    )
                )
                print('split val loss: onset {:.4f} offset {:.4f} pitch octave {:.4f} pitch class {:.4f}'.format(
                        split_val_loss[0]/len(self.valid_loader),
                        split_val_loss[1]/len(self.valid_loader),
                        split_val_loss[2]/len(self.valid_loader),
                        split_val_loss[3]/len(self.valid_loader)
                    )
                )
        print('Training done in {:.1f} minutes.'.format((time.time()-start_time)/60))

    def _parse_frame_info(self, frame_info, onset_thres, offset_thres):
        """Parse frame info [(onset_probs, offset_probs, pitch_class)...] into desired label format."""

        result = []
        current_onset = None
        pitch_counter = []

        last_onset = 0.0
        onset_seq = np.array([frame_info[i][0] for i in range(len(frame_info))])

        local_max_size = 3
        current_time = 0.0

        onset_seq_length = len(onset_seq)

        for i in range(len(frame_info)):

            current_time = FRAME_LENGTH*i
            info = frame_info[i]

            backward_frames = i - local_max_size
            if backward_frames < 0:
                backward_frames = 0

            forward_frames = i + local_max_size + 1
            if forward_frames > onset_seq_length - 1:
                forward_frames = onset_seq_length - 1

            # local max and more than threshold
            if info[0] >= onset_thres and onset_seq[i] == np.amax(onset_seq[backward_frames : forward_frames]):

                if current_onset is None:
                    current_onset = current_time
                    last_onset = info[0] - onset_thres

                else:
                    if len(pitch_counter) > 0:
                        result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])

                    current_onset = current_time
                    last_onset = info[0] - onset_thres
                    pitch_counter = []

            elif info[1] >= offset_thres:  # If is offset
                if current_onset is not None:
                    if len(pitch_counter) > 0:
                        result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])
                    current_onset = None

                    pitch_counter = []

            # If current_onset exist, add count for the pitch
            if current_onset is not None:
                final_pitch = int(info[2]* 12 + info[3])
                if info[2] != 4 and info[3] != 12:
                # if final_pitch != 60:
                    pitch_counter.append(final_pitch)

        if current_onset is not None:
            if len(pitch_counter) > 0:
                result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])
            current_onset = None

        return result

    def predict(self, test_dataset, results={}, onset_thres=0.1, offset_thres=0.5):
        """Predict results for a given test dataset."""
        # Setup params and dataloader
        batch_size = 500
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )

        # Start predicting
        my_sm = torch.nn.Softmax(dim=0)
        self.model.eval()
        with torch.no_grad():
            song_frames_table = {}
            raw_data = {}
            for batch_idx, batch in enumerate(tqdm(test_loader)):
                # Parse batch data
                input_tensor = batch[0].to(self.device)
                song_ids = batch[1]

                result_tuple = self.model(input_tensor)
                onset_logits = result_tuple[0]
                offset_logits = result_tuple[1]
                pitch_octave_logits = result_tuple[2]
                pitch_class_logits = result_tuple[3]

                onset_probs, offset_probs = torch.sigmoid(onset_logits).cpu(), torch.sigmoid(offset_logits).cpu()
                pitch_octave_logits, pitch_class_logits = pitch_octave_logits.cpu(), pitch_class_logits.cpu()
                # print (pitch_octave_logits)


                # Collect frames for corresponding songs
                for bid, song_id in enumerate(song_ids):
                    frame_info = (onset_probs[bid], offset_probs[bid], torch.argmax(pitch_octave_logits[bid]).item()
                            , torch.argmax(pitch_class_logits[bid]).item())

                    song_frames_table.setdefault(song_id, [])
                    song_frames_table[song_id].append(frame_info)
                        

            # Parse frame info into output format for every song
            for song_id, frame_info in song_frames_table.items():
                results[song_id] = self._parse_frame_info(frame_info, onset_thres=onset_thres, offset_thres=offset_thres)
        return results
