#Import Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio

import os
import numpy as np
import matplotlib.pyplot as plt

#Extra functions for calculating loss
def avg_wer(wer_scores, combined_ref_len):
    return float(sum(wer_scores)) / float(combined_ref_len)


def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0,n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Levenshtein distance and word number of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)
    
    #print(ref_words, hyp_words)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Levenshtein distance and length of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level. WER is defined as:
    .. math::
        WER = (Sw + Dw + Iw) / Nw
    where
    .. code-block:: text
        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference
    We can use levenshtein distance to calculate WER. Please draw an attention
    that empty items will be removed when splitting sentences by delimiter.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Word error rate.
    :rtype: float
    :raises ValueError: If word number of reference is zero.
    """
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
                                         delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """Calculate charactor error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
    .. math::
        CER = (Sc + Dc + Ic) / Nc
    where
    .. code-block:: text
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
    We can use levenshtein distance to calculate CER. Chinese input should be
    encoded to unicode. Please draw an attention that the leading and tailing
    space characters will be truncated and multiple consecutive space
    characters in a sentence will be replaced by one space character.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Character error rate.
    :rtype: float
    :raises ValueError: If the reference length is zero.
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer

###--------------------------------------------------------------------------------------------

#Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"

#Download dataset
dataset = torchaudio.datasets.SPEECHCOMMANDS(root="C:/Users/kevin/Pytorch/dataset",
                                            url="speech_commands_v0.02",
                                            download=False)

dataset_directory = 'C:/Users/kevin/Pytorch/dataset/SpeechCommands/speech_commands_v0.02/'

#Get all labels
labels_list = os.listdir(dataset_directory)

#Remove labels that are not actual classes
labels_list.remove('LICENSE')
labels_list.remove('README.md')
labels_list.remove('validation_list.txt')
labels_list.remove('.DS_Store')
labels_list.remove('testing_list.txt')
#labels_list.remove('_background_noise_') #Need a ground truth label for noise as well

#Create a mapping list
characters_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
            'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
            's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '<SPACE>']

class TextProcessing:
    def __init__(self, characters):
        if characters != None:
            #For transcription recognization
            self.characters = characters
            self.characters_map = dict()
            self.index_characters_map = dict()
            for i, character in enumerate(self.characters):
                self.characters_map[character] = i
                self.index_characters_map[i] = character
        
    def text_to_int(self, text):
        seq = []
        for ch in text:
            seq.append(self.characters_map[ch])
        return seq
    
    def int_to_text(self, seq):
        string = []
        for i in seq:
            string.append(self.index_characters_map[i])
        return ''.join(string)

#Initialize text processing        
textprocessing = TextProcessing(characters_list)

#MelSpectogram is used here. However, MFCCs could just as easily be used here.
train_audio_transforms = nn.Sequential(
    #torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64),
    torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=64),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=2),
    torchaudio.transforms.TimeMasking(time_mask_param=5)
    )

val_audio_transforms = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=64)

def data_processing(data, data_type="train"):
    tensors = []
    targets = []
    input_len = []
    targets_len = []   
    
    for waveform, _, label, *_ in data:
        if data_type=="train":
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1) #[81, 64] shape
        else:
            spec = val_audio_transforms(waveform).squeeze(0).transpose(0, 1) #[81, 64] shape
        
        tensors.append(spec)
        label = torch.Tensor(textprocessing.text_to_int(label))
        targets.append(label)
        input_len.append(spec.shape[0]//2)
        targets_len.append(len(label))
    
    #Padding tensors
    tensors = nn.utils.rnn.pad_sequence(tensors,
                                        batch_first=True,
                                        padding_value=26).unsqueeze(1).transpose(2, 3)
    
    #Padding targets with 26 value because that is the noise index
    targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=26)
    
    return tensors, targets, input_len, targets_len

#Split dataset 80/20
train_data, test_data = random_split(dataset, lengths=[round(len(dataset)*0.8),
                                                    round(len(dataset)*0.2)])

train_loader = DataLoader(train_data, batch_size=100, shuffle=True, collate_fn=lambda x: data_processing(x, 'train'))

test_loader = DataLoader(test_data, batch_size=100, shuffle=True, collate_fn=lambda x: data_processing(x, 'test'))

class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        #Why transpose for layer norm? Find out...
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 

class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()
        
        self.cnn1 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)
    
class BidirectionalLSTM(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalLSTM, self).__init__()
        
        self.lstm = nn.LSTM(rnn_dim, hidden_size, num_layers=1, batch_first=batch_first,
                            bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        return x
    
class SpeechRecognition(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats,
                 rnn_dim, hidden_size, batch_first, n_classes):
        super(SpeechRecognition, self).__init__()
        
        self.hcnn = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=3//2)
        
        self.cnn = ResidualCNN(out_channels, out_channels, kernel, stride, dropout, n_feats)
        self.fc = nn.Linear(out_channels**2, rnn_dim)
        self.lstm = BidirectionalLSTM(rnn_dim, hidden_size, dropout, batch_first)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2, rnn_dim),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(rnn_dim, n_classes)
            )
        
    def forward(self, x):
        x = self.hcnn(x)
        x = self.cnn(x)
        #print(x.shape)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)
        #print(x.shape)
        x = self.fc(x)
        x = self.lstm(x)
        x = self.classifier(x)
        #print(x.shape)
        return x
        
def GreedyDecoder(output, labels, label_lengths, blank_label=26, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(textprocessing.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(textprocessing.int_to_text(decode))
    return decodes, targets

model = SpeechRecognition(in_channels=1,
                          out_channels=64,
                          kernel=3, stride=1, dropout=0.2,
                          n_feats=64,
                          rnn_dim=256,
                          hidden_size=100,
                          batch_first=True,
                          n_classes=27).to(device)  

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
	max_lr=5e-4,
	steps_per_epoch=int(len(train_loader)),
	epochs=20,
	anneal_strategy='linear')

loss_fn = nn.CTCLoss(blank=26).to(device)         
epochs=20   

def train(dataloader, model, optimizer, loss_fn, scheduler):
    size = len(dataloader.dataset)
    model.train()
    
    for batch, _data in enumerate(dataloader):
        tensors, targets, input_len, targets_len = _data
        tensors, targets = tensors.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        pred = model(tensors) # (batch, time, n_class)
        pred = F.log_softmax(pred, dim=2)
        pred = pred.transpose(0, 1) # (time, batch, n_class)
        #Necessary for loss function to have prediction in this shape
        
        loss = loss_fn(pred, targets, input_len, targets_len)
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(tensors)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
def test(dataloader, model, optimizer, loss_fn):
    size = len(dataloader)
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    
    with torch.no_grad():
        for batch, _data in enumerate(dataloader):
            tensors, targets, input_len, targets_len = _data
            tensors, targets = tensors.to(device), targets.to(device)
            
            pred = model(tensors) # (batch, time, n_class)
            pred = F.log_softmax(pred, dim=2)
            pred = pred.transpose(0, 1) # (time, batch, n_class)
            
            loss = loss_fn(pred, targets, input_len, targets_len)
            test_loss += loss.item()/size
            
            decoded_preds, decoded_targets = GreedyDecoder(pred.transpose(0, 1), targets, targets_len)
            for j in range(len(decoded_preds)):
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))
                
    avg_cer = sum(test_cer)/len(test_cer)
    avg_wer = sum(test_wer)/len(test_wer)
    print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))

for epoch in range(1, epochs+1):
    print(f"Epoch {epoch}\n-------------------------------")
    train(train_loader, model, optimizer, loss_fn, scheduler)
    test(test_loader, model, optimizer, loss_fn)

#Loading a single batch
count=0
with torch.no_grad():
    for batch, _data in enumerate(test_loader):
        if count>=1:
            break
        count+=1
        tensors, targets, input_len, targets_len = _data
        tensors, targets = tensors.to(device), targets.to(device)

        pred = model(tensors)
        val = targets
        
        print(pred)
        print(val)
        
com = torch.argmax(pred, dim=2)

#print(com[0], val[0])

#Printing Output
#Background noise is classified as <SPACE> token!
command=""
for i in com[0]:
    if int(i.item())!=26:
        command+=characters_list[int(i.item())]

print(command)

#Plot and save the model
plt.plot(tensors[0].squeeze(0).cpu().numpy())
plt.title("Waveform")

PATH = ""
torch.save(model.state_dict(), os.path.join(PATH,"third_model.pt"))

