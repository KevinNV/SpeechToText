# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 00:14:27 2022

@author: kevin
"""
import torch.nn as nn
import torch.nn.functional as F

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
        
        #self.cnn0 = nn.Conv2d(in_channels, out_channels, kernel, stride)
        self.cnn1 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        #x = self.cnn0(x)
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