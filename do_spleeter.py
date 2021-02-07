import youtube_dl
import os
import sys
import json
import librosa
import numpy as np

if __name__ == '__main__':

    yt_id = []
    data_id = []
    offset_list = []

    dataset_dir = sys.argv[1]

    from spleeter.separator import Separator
    import warnings
    separator = Separator('spleeter:2stems')

    for the_dir in os.listdir(dataset_dir):
        mix_path = os.path.join(dataset_dir, the_dir, "Mixture.mp3")
    
        y, sr = librosa.core.load(mix_path, sr=None, mono=True)
        if sr != 44100:
            y = librosa.core.resample(y=y, orig_sr=sr, target_sr=44100)

        waveform = np.expand_dims(y, axis=1)

        prediction = separator.separate(waveform)
        voc = librosa.core.to_mono(prediction["vocals"].T)
        voc = np.clip(voc, -1.0, 1.0)

        acc = librosa.core.to_mono(prediction["accompaniment"].T)
        acc = np.clip(acc, -1.0, 1.0)

        import soundfile
        soundfile.write(os.path.join(dataset_dir, the_dir, "Vocal.wav"), voc, 44100, subtype='PCM_16')
        soundfile.write(os.path.join(dataset_dir, the_dir, "Inst.wav"), acc, 44100, subtype='PCM_16')
