# SpeechToText
---
Speech to text conversion (Transcript generation) using PyTorch with CNNs and LSTMs.<br>

# Table of Contents
---
+ **Method/Process**
+ **Status**
+ **Future Tasks**

# Method/Process
---
Step 1: Download SpeechCommands dataset from PyTorch datasets. We will use this dataset for now since I had started creating this project as a classifier first and then converted the classifier into a STT model. However, here, the main focus will not be the classifier model but the STT model.<br>
Step 2: Firstly, we will convert the raw audio (.wav) to Spectograms or MFCCs (Mel-Frequency Cepstral Coefficients) in this case.<br>
Step 3: Apply CNNs to the MFCCs to extract features.<br>
Step 4: Apply Bidirectional LSTM layers to CNN outputs.<br>
Step 5: Save model as .pt file and use it to generate transcriptions where necessary!

# Status
---
**Completed** <br>
Link: [Full Project](https://github.com/Begelit/Speech-To-Text-Web-App-Django?tab=readme-ov-file)

# Future Tasks
---
Task 1: Apply the same method on LibriSpeech dataset for better generalization.<br>
Task 2: Apply various types of augmentations.<br>
Task 3: Apply Transformer instead of LSTM for STT.<br>
