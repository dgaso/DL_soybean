# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 10:01:05 2023

@author: cuel001
"""

import torch
import torch.nn as nn

from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer



class TransformerModel(nn.Module):
    def __init__(self, c_in, c_out, d_model=64, n_head=1, d_ffn=128, dropout=0.1, 
                 activation="relu", n_layers=1, sigm_out = False):
        """
        Args:
            c_in: the number of features (aka variables, dimensions, channels) in the time series dataset
            c_out: the number of target classes
            d_model: total dimension of the model.
            nhead:  parallel attention heads.
            d_ffn: the dimension of the feedforward network model.
            dropout: a Dropout layer on attn_output_weights.
            activation: the activation function of intermediate layer, relu or gelu.
            num_layers: the number of sub-encoder-layers in the encoder.
            
        Input shape:
            bs (batch size) x nvars (aka variables, dimensions, channels) x seq_len (aka time steps)
            """
        super(TransformerModel, self).__init__()
        
        self.sigm_out = sigm_out
        self.inlinear = nn.Linear(c_in, d_model)
        self.relu = nn.ReLU()
        encoder_layer = TransformerEncoderLayer(d_model, n_head, dim_feedforward=d_ffn, dropout=dropout, activation=activation)
        encoder_norm = nn.LayerNorm(d_model)        
        self.transformer_encoder = TransformerEncoder(encoder_layer, n_layers, norm=encoder_norm)
        self.outlinear = nn.Linear(d_model, c_out)
        
        self.sigmoid = nn.Sigmoid()
        
        
        
    def forward(self,x):
        x = x.permute(2, 0, 1) # bs x nvars x seq_len -> seq_len x bs x nvars
        x = self.inlinear(x) # seq_len x bs x nvars -> seq_len x bs x d_model
        x = self.relu(x)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 0) # seq_len x bs x d_model -> bs x seq_len x d_model
        x = x.max(1, keepdim=False)[0]
        x = self.relu(x)
        x = self.outlinear(x)
        if self.sigm_out:
            x = self.sigmoid(x)
        return x
    
    
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initialize hidden and cell states
        #h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device) # multiplied by 2 for bidirectionality
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device) # multiplied by 2 for bidirectionality


        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        if self.sigm_out:
            out = self.sigmoid(out)
        return out
    

#Define the LSTM model

class _RNN_Base(nn.Module):
    def __init__(self, c_in, c_out, hidden_size=100, n_layers=1, bias=True, 
                 rnn_dropout=0, bidirectional=False, fc_dropout=0., 
                 init_weights=True, sigm_out = False):
        super(_RNN_Base, self).__init__()
        self.sigm_out = sigm_out
        self.rnn = self._cell(c_in, hidden_size, num_layers=n_layers, bias=bias, batch_first=True, dropout=rnn_dropout, 
                              bidirectional=bidirectional)
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout else nn.Identity()
        self.fc = nn.Linear(hidden_size * (1 + bidirectional), c_out)
        if init_weights: self.apply(self._weights_init)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): 
        x = x.transpose(2,1)    # [batch_size x n_vars x seq_len] --> [batch_size x seq_len x n_vars]
        output, _ = self.rnn(x) # output from all sequence steps: [batch_size x seq_len x hidden_size * (1 + bidirectional)]
        output = output[:, -1]  # output from last sequence step : [batch_size x hidden_size * (1 + bidirectional)]
        output = self.fc(self.dropout(output))
        if self.sigm_out:
            output = self.sigmoid(output)
        return output
    
    def _weights_init(self, m): 
        # same initialization as keras. Adapted from the initialization developed 
        # by JUN KODA (https://www.kaggle.com/junkoda) in this notebook
        # https://www.kaggle.com/junkoda/pytorch-lstm-with-tensorflow-like-initialization
        for name, params in m.named_parameters():
            if "weight_ih" in name: 
                nn.init.xavier_normal_(params)
            elif 'weight_hh' in name: 
                nn.init.orthogonal_(params)
            elif 'bias_ih' in name:
                params.data.fill_(0)
                # Set forget-gate bias to 1
                n = params.size(0)
                params.data[(n // 4):(n // 2)].fill_(1)
            elif 'bias_hh' in name:
                params.data.fill_(0)
        
class RNN(_RNN_Base):
    _cell = nn.RNN
    
class LSTM(_RNN_Base):
    _cell = nn.LSTM
    
class GRU(_RNN_Base):
    _cell = nn.GRU
    
    
class TimeSeriesCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kernel_size, n_layers = 1,
                 fc_dropout=0., sigm_out = False, time=None):
        super(TimeSeriesCNN, self).__init__()
        self.num_conv_layers = n_layers
        self.conv_layers = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.sigm_out = sigm_out

        out_channels = hidden_size
        for i in range(n_layers):
            conv_layer = nn.Conv1d(in_channels=input_size if i == 0 else out_channels*i, out_channels=out_channels+(out_channels*i), 
                                   kernel_size=3, padding=1)
            self.conv_layers.append(conv_layer)


        # Adjust the size for the fully connected layer based on your convolution and pooling layers
        self.fc_input_size = out_channels*n_layers * (time//(2**n_layers))
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout else nn.Identity()

        # Sigmoid activation for binary classification        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        for i in range(self.num_conv_layers):
            x = self.conv_layers[i](x)
            x = self.relu(x)
            x = self.pool(x)
        
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.relu(self.fc1(x))
        x = self.fc2(self.dropout(x))
        if self.sigm_out:
            # Apply sigmoid activation
            x = self.sigmoid(x)
        return x
    

class TemporalConvolutionalNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kernel_size, n_layers = 1,
                 fc_dropout=0., sigm_out = False):
        super(TemporalConvolutionalNetwork, self).__init__()
        
        self.sigm_out = sigm_out
        self.n_layers = n_layers
        # 1D Convolutional layer
        self.conv1d = nn.Conv1d(in_channels=input_size,
                                out_channels=hidden_size, kernel_size=kernel_size//n_layers)
        
        self.conv2d = nn.Conv1d(in_channels=hidden_size,
                                out_channels=hidden_size*2, kernel_size=kernel_size//n_layers)
        
        self.conv3d = nn.Conv1d(in_channels=hidden_size*2,
                                out_channels=hidden_size, kernel_size=kernel_size//n_layers)


        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout else nn.Identity()

        # Sigmoid activation for binary classification        
        self.sigmoid = nn.Sigmoid()
        
        self.relu = nn.ReLU()
        

    def forward(self, x):
        # Input x should have shape (batch_size, temporal_dim, input_channels)

        # Apply 1D convolution
        x = self.conv1d(x)
        x = self.relu(x)
        
        if self.n_layers >= 2:
            x = self.conv2d(x)
            x = self.relu(x)
            
            if self.n_layers == 3:
                x = self.conv3d(x)
                x = self.relu(x)

        # Flatten the output from convolutional layer
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.fc(self.dropout(x))
        
        if self.sigm_out:
            # Apply sigmoid activation
            x = self.sigmoid(x)
        

        return x
