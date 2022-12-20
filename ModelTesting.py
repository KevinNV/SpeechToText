# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 22:42:31 2022

@author: kevin
"""

import torch
import torchaudio
import torch.nn as nn
from SpeechModelOnly import SpeechRecognition
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SpeechRecognition(in_channels=1,
                          out_channels=64,
                          kernel=3, stride=1, dropout=0.2,
                          n_feats=64,
                          rnn_dim=256,
                          hidden_size=100,
                          batch_first=True,
                          n_classes=27).to(device)

model.load_state_dict(torch.load("model.pt")) #Load model
model.eval() #Set to eval state

#Check metadata
PATH = "Dog.wav"
metadata = torchaudio.info(PATH)
print(metadata)

#Load audio as waveform
waveform, sample_rate = torchaudio.load(PATH)

waveform = waveform[:, :48000] #Just for my satisfaction, clipping it

#Plot to confirm length
plt.plot(waveform.t().numpy())
plt.show()


#label = torch.Tensor([20, 15, 26, 26, 26, 26, 26, 26]) #Up label
#label = torch.Tensor([18, 7, 4, 8, 11, 0, 26, 26]) #Sheila label
label = torch.Tensor([3, 14, 6, 26, 26, 26, 26, 26]) #Dog label
label_text = "Dog"
label_len = len(label)
label = label.to(device)

#Resampling
new_sample_rate = 16000
transform = nn.Sequential(
    torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate),
    torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=64)
    )

channel=0
waveform = transform(waveform[channel, :].view(1, -1)).unsqueeze(0)

#Predicting
pred = model(waveform.to(device))
com = torch.argmax(pred, dim=2)

#Create a mapping list
characters_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
            'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
            's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '<SPACE>']

command=""
for i in com[0]:
    if int(i.item())!=26:
        command+=characters_list[i]

print(command)