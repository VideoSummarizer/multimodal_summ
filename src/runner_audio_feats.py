from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
df = pd.read_csv('video_map.csv')

for index,row in tqdm(df.iterrows(),total=df.shape[0]):
    os.system('rm audio/* video/*')
    os.system('cp '+row['video_path']+' video/')
    file_name = row['video_path'].split('/')[-1]
    os.system('ffmpeg -i video/'+file_name+'  -ar 16000 -ac 1  audio/outfile.wav')
    (rate,sig) = wav.read("audio/outfile.wav")
    mfcc_feat = mfcc(sig,rate)
    fbank_feat = logfbank(sig,rate)

    feats = fbank_feat[1:3,:]
    np.save(row['feature_path'],feats.flatten())
