# Singing Transcription
This repo contains four parts of code/resources. All of them are related to the paper:
##### Jun-You Wang and Jyh-Shing Roger Jang, "On the Preparation and Validation of a Large-scale Dataset of Singing Transcription", ICASSP2021 accepted paper.
Note that the main purpose of creating this repo is for reproducibility of this paper, so the code may lack some flexibility.

### MIR-ST500_20210206
MIR-ST500 dataset is a singing transcription dataset proposed in this paper. It consists of 500 pop songs. For each song, we provide the label of notes (vocal part) and the correponding Youtube URL. You can download these songs using the provided script (get_youtube.py).

For more information, please refer to "MIR-ST500_20210206/Readme".

#### Update History
2021.02.07 Upload the first version of MIR-ST500 dataset.<br>
2021.04.08 Fix the note parsing issue, regenerate "MIR-ST500_corrected.json" from MIDI files.<br>
2021.05.23 Fix note overlapping issue.<br>

### adjust_onset
Source code used to refine onset labels automatically.

### evaluate
We use evaluate/evaluate.py to evaluate our singing transcription model.<br>
"val_1005_55_icassp_ver.json" is the result transcribed by our model.<br>
However, after submitting the manuscript to ICASSP2021, we found out that there is a bug in the post-processing code, so you may not be able to reproduce the exactly same result using our python scripts and the pre-trained model (in "AST" folder). However, the result should still be really close to "val_1005_55_icassp_ver.json".

### AST
This folder contains everything needed to reproduce our result:<br>
(1) The pre-trained singing transcription model (EfficientNet-b0).<br>
(2) The code to train a singing transcription model.<br>
(3) The code that uses a pre-trained model to predict songs.

## Sample Usage

### Inference
> pip install -r requirements.txt<br>
> cd AST<br>
> python do_everything.py [input_audio] [output_midi_path] -p model/1005_e_4 -s -on 0.4 -off 0.5<br>

The script do_everything.py will do everything.<br>
-p: Model path.<br>
-s: Do SVS (singing voice separation, in this script, we use Spleeter (https://github.com/deezer/spleeter) to perform SVS) or not.<br>
-on: Onset threshold. The higher it is, the lower the notes number that will be transcribed.<br>
-off: Silence threshold. It decides where offsets are placed.<br>

### Training a singing transcription model from scratch

> python get_youtube.py MIR-ST500_20210206/MIR-ST500_link.json train test

This will download 500 songs from Youtube automatically. Song id #1~#400 (training set) will be saved to "train/", #401~#500(test set) will be saved to "test/"

> python do_spleeter.py train/<br>
> python do_spleeter.py test/

And then, we have to use an SVS program to extract vocal, and write the vocal file to "Vocal.wav". Here, do_spleeter.py uses Spleeter to do the job.

> cd AST<br>
> python generate_dataset.py ../train ../MIR-ST500_20210206/MIR-ST500_corrected.json ./ .pkl<br>
> python generate_dataset.py ../test ../MIR-ST500_20210206/MIR-ST500_corrected.json ./ .pkl

This will generate two datasets, "train.pkl" and "test.pkl".<br>
Using these two dataset files, we can then train a singing transcription model.

> python train.py train.pkl test.pkl models/ effnet cuda:0

The script will then use train.pkl as training set, test.pkl as validation set (it will predict the validation set every epoch, and print the validation loss in the end of every epoch). If cuda is available, the model will be trained using "cuda:0".<br>
In the end of each epoch, a model file called called "effnet_{epoch}" will be saved at "models/".<br>
Using one NVIDIA GTX 1080Ti GPU, it may take up to 2 hours to complete one epoch. We trained four epochs (about 300K steps) to obtain the pre-trained model "model/1005_e_4".

And that's all!

### Conclusion?
This repo is still an ongoing project. The dataset itself may be revised in the future. If you have any question, feel free to contact me directly, or open an issue on Github!
