import sys
import os
import time
import argparse
import torch
from predictor import EffNetPredictor

import argparse
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # nopep8
from data_utils.seq_dataset import SeqDataset
from pathlib import Path
from tqdm import tqdm
import mido
import warnings
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

def notes2mid(notes):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    mid.ticks_per_beat = 480
    new_tempo = mido.bpm2tempo(120.0)

    track.append(mido.MetaMessage('set_tempo', tempo=new_tempo))
    track.append(mido.Message('program_change', program=0, time=0))

    cur_total_tick = 0


    for note in notes:
        if note[2] == 0:
            continue
        note[2] = int(round(note[2]))

        ticks_since_previous_onset = int(mido.second2tick(note[0], ticks_per_beat=480, tempo=new_tempo))
        ticks_current_note = int(mido.second2tick(note[1]-0.0001, ticks_per_beat=480, tempo=new_tempo))
        note_on_length = ticks_since_previous_onset - cur_total_tick
        note_off_length = ticks_current_note - note_on_length - cur_total_tick

        track.append(mido.Message('note_on', note=note[2], velocity=100, time=note_on_length))
        track.append(mido.Message('note_off', note=note[2], velocity=100, time=note_off_length))
        cur_total_tick = cur_total_tick + note_on_length + note_off_length

    return mid
    

def convert_to_midi(predicted_result, song_id, output_path):
    to_convert = predicted_result[song_id]
    mid = notes2mid(to_convert)
    mid.save(output_path)

def predict_one_song(predictor, wav_path, song_id, results, do_svs, tomidi, output_path, onset_thres, offset_thres):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    test_dataset = SeqDataset(wav_path, song_id, do_svs=do_svs)

    results = predictor.predict(test_dataset, results=results, onset_thres=onset_thres, offset_thres=offset_thres)
    if tomidi:
        convert_to_midi(results, song_id, output_path)
    return results


def predict_whole_dir(predictor, test_dir, do_svs, output_json_path, onset_thres, offset_thres):
    
    results = {}
    for song_dir in sorted(Path(test_dir).iterdir()):
        wav_path = str(song_dir / 'Vocal.wav')
        song_id = song_dir.stem

        if not os.path.isfile(wav_path):
            continue

        results = predict_one_song(predictor, wav_path, song_id, results, do_svs=do_svs
            , tomidi=False, output_path=None, onset_thres=float(args.onset_thres), offset_thres=float(args.offset_thres))

    with open(output_json_path, 'w') as f:
        output_string = json.dumps(results)
        f.write(output_string)


def main(args):
    model_path = args.model_path
    input_path = args.input
    output_path = args.output

    device= "cpu"
    if torch.cuda.is_available():
        device = args.device
    # print ("use", device)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    my_predictor = EffNetPredictor(device=device, model_path=args.model_path)
    song_id = '1'
    results = {}
    do_svs = args.svs

    print('Forwarding model...')

    if os.path.isfile(input_path):
        predict_one_song(my_predictor, input_path, song_id, results, do_svs=do_svs
            , tomidi=True, output_path=output_path, onset_thres=float(args.onset_thres), offset_thres=float(args.offset_thres))

    elif os.path.isdir(input_path):
        predict_whole_dir(my_predictor, input_path, do_svs=do_svs
            , output_json_path=output_path, onset_thres=float(args.onset_thres), offset_thres=float(args.offset_thres))
    else:
        print ("\"input\" argument is not a valid path/directory, no audio is trancribed......")


if __name__ == '__main__':
    """
    "input" argument can be an audio file or a folder.
    If "input" is an audio, the program simply transcribe it, and then write the resulted MIDI file to "output".
    If "input" is a directory, it should contain several subdirectories. Each subdirectory should contains an audio file called "Vocal.wav". 
    The program will transcribe all "Vocal.wav" files, and write the resulted JSON file
    (not MIDI file, since there are multiple audio files to transcribe) to "output".

    "-p" argement specifies the model path. Default is "models/1005_e_4", which is the (EfficientNet-b0) model we used in our paper. 
    "-s" argument decides if Spleeter (SVS program) should be used or not.
    "-on" argement specifies the onset threshold, while "-off" argement specifies the silence threshold.
    "-d" argement specifies the device (e.g. cuda:0) to use if cuda is available.
    If you specify "-d cpu", that means you disable the use of gpu even if cuda is available.
    """

    # print (time.time())
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="input audio/folder path")
    parser.add_argument('output', help="output MIDI/JSON path")
    parser.add_argument('-p', '--model_path', default="models/1005_e_4", help="model path")
    parser.add_argument('-s', '--svs', action="store_true", help="use Spleeter to extract vocal or not")
    parser.add_argument('-on', "--onset_thres", default=0.4, help="onset threshold")
    parser.add_argument('-off', "--offset_thres", default=0.5, help="silence threshold")
    parser.add_argument('-d', "--device", default="cuda:0", help="device to use if cuda is available")

    args = parser.parse_args()

    main(args)
    # print (time.time())
