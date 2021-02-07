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

from MyDNN import MyDNN
import math

FRAME_LENGTH = librosa.frames_to_time(1, sr=16000, hop_length=320)

class UnvoicedPredictor:
    def __init__(self, device= "cuda:0", model_path=None):
        """
        Params:
        model_path: Optional pretrained model file
        """
        # Initialize model
        self.device = device
        self.model = MyDNN().to(self.device)

        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location= self.device))
            print('Model read from {}.'.format(model_path))

        print('Predictor initialized.')


    def fit(self, train_dataset_path, valid_dataset_path, model_dir, **training_args):

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
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.1,], device="cuda:0"))
        
        # Read the datasets
        print('Reading datasets...')
        print ('cur time: %.6f' %(time.time()))
        with open(self.train_dataset_path, 'rb') as f:
            self.training_dataset = pickle.load(f)
        with open(self.valid_dataset_path, 'rb') as f:
            self.validation_dataset = pickle.load(f)

        cnt = 0
        # print (len(self.training_dataset.data_instances[0][0][0]))

        # Setup dataloader and initial variables
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
        training_loss_list = []
        valid_loss_list = []

        # Start training
        print('Start training...')
        print ('cur time: %.6f' %(time.time()))
        self.iters_per_epoch = len(self.train_loader)
        print (len(self.train_loader))

        for epoch in range(1, self.epoch + 1):
            self.model.train()

            # Run iterations
            total_training_loss = 0
            total_split_loss = np.zeros(4)
            # total_length = 0
            for batch_idx, batch in enumerate(self.train_loader):
                self.model.train()
                # Parse batch data
                input_tensor = batch[0].unsqueeze(1).to(self.device)
                label = batch[1].unsqueeze(1).float().to(self.device)

                loss = 0
                length = len(input_tensor)
                
                self.optimizer.zero_grad()
                result = self.model(input_tensor)

                loss = self.criterion(result, label)
                loss.backward()
                self.optimizer.step()
                total_training_loss += loss.item()
                # Free GPU memory
                torch.cuda.empty_cache()

            if epoch % self.save_every_epoch == 0:
                # Perform validation
                self.model.eval()
                with torch.no_grad():
                    total_valid_loss = 0
                    split_val_loss = np.zeros(4)
                    for batch_idx, batch in enumerate(self.valid_loader):
                        # Parse batch data
                        input_tensor = batch[0].unsqueeze(1).to(self.device)
                        label = batch[1].unsqueeze(1).float().to(self.device)
                        
                        result = self.model(input_tensor)

                        loss = self.criterion(result, label)
                        total_valid_loss += loss.item()

                # Save model
                save_dict = self.model.state_dict()
                target_model_path = Path(self.model_dir) / 'dnn_e_{}'.format(epoch)
                torch.save(save_dict, target_model_path)

                # Save loss list
                training_loss_list.append((epoch, total_training_loss/len(self.train_loader)))
                valid_loss_list.append((epoch, total_valid_loss/len(self.valid_loader)))

                # Epoch statistics
                print(
                    '| Epoch [{:4d}/{:4d}] Train Loss {:.4f} Valid Loss {:.4f} Time {:.1f}'.format(
                        epoch,
                        self.epoch,
                        training_loss_list[-1][1],
                        valid_loss_list[-1][1],
                        time.time()-start_time,
                    )
                )

        print('Training done in {:.1f} minutes.'.format((time.time()-start_time)/60))

    def compute_viterbi(self, log_prob, transition):
        log_transition = np.log(np.array(transition))
        cur_prob = np.zeros(2)
        cur_seq = [[0,], [1,]]

        for i in range(len(log_prob)):
            new_prob = np.zeros(2)
            new_seq = []

            for j in range(len(cur_prob)):
                prob = np.zeros(2)
                for k in range(len(cur_prob)):
                    prob[k] = cur_prob[k] + log_transition[k][j] + log_prob[i][j]
                # print (prob)
                # print (np.argmax(prob))
                new_seq.append(list(cur_seq[np.argmax(prob)]))
                new_seq[-1].append(j)
                new_prob[j] = np.amax(prob)

            cur_prob = np.copy(new_prob)
            cur_seq = list(new_seq)

        return cur_seq[np.argmax(cur_prob)][1:]



    def _parse_frame_info(self, frame_info, prior_prob):
        result = []

        frame_info = torch.tensor(frame_info)

        local_max_size = 2
        current_time = 0.0
        current_onset = None

        # This transition probability is computed from the training set(900 clips) of MIR-1k
        transition = [[23696.0/(23696.0+5893.0), 5893.0/(23696.0+5893.0)], [5902.0/(5902.0+362365.0), 362365.0/(5902.0+362365.0)]]
        # transition = [[0.5, 0.5], [0.5, 0.5]]

        info = nn.functional.logsigmoid(torch.cat((-frame_info.unsqueeze(1), frame_info.unsqueeze(1)), dim=1)).numpy()
        for i in range(len(info)):
            info[i][1] = info[i][1] - prior_prob
        sequence = self.compute_viterbi(info, transition)
        # print (info)
        # sequence = []
        for i in range(len(frame_info)):

            current_time = FRAME_LENGTH*i + FRAME_LENGTH / 2.0

            if sequence[i] == 0:
                if current_onset is None:
                    current_onset = current_time

            else:  # If is offset
                if current_onset is not None:
                    result.append([current_onset, current_time])
                    current_onset = None


        if current_onset is not None and current_onset != current_time:
            result.append([current_onset, current_time])
            current_onset = None

        return result, sequence

    def predict_test(self, test_dataset, results= {}, show_tqdm=True, prior_prob=-1.16):
        """Predict results for a given test dataset."""
        # Setup params and dataloader
        batch_size = 1000
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )

        # Start predicting
        # my_sm = torch.nn.Softmax(dim=0)
        self.model.eval()
        sequences = {}
        with torch.no_grad():
            song_frames_table = {}

            print('Forwarding model...')
            for batch_idx, batch in tqdm(enumerate(test_loader)):
                # Parse batch data
                input_tensor = batch[0].unsqueeze(1).to(self.device)
                song_ids = batch[1]

                result = self.model(input_tensor)

                for bid, song_id in enumerate(song_ids):
                    frame_info = result[bid]
                    song_frames_table.setdefault(song_id, [])
                    song_frames_table[song_id].append(frame_info)
                        
            
            # Parse frame info into output format for every song
            for song_id, frame_info in song_frames_table.items():
                # print (song_id)
                results[song_id], sequences[song_id] = self._parse_frame_info(frame_info, prior_prob=prior_prob)

        return results, sequences
