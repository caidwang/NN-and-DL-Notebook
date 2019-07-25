# implement GRU in pytorch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=False):
        '''
        Create a GRU cell.
        
        bias -- Not implement yet.
        '''
        super(GRUCell, self).__init__()
        self.Wx = nn.Parameters(torch.randn(input_size, hidden_size))
        self.Wh = nn.Parameters(torch.randn(hidden_size, hidden_size))
        self.input_to_hidden = nn.Linear(input_size, hidden_size * 2)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size * 2)
        
    def forward(self, xt, h_prevt):
        '''
        Do forward calculation.
        
        Arguments:
        xt -- input of t step, (batch, input_size)
        h_prevt -- hidden state of previous time step, (batch, hidden_size)
        
        Returns:
        ht -- hidden state of current time step, (batch, hidden_size)
        '''
        gates = self.input_to_hidden(xt) + self.hidden_to_hidden(h_prevt) # linear transform in zt, rt (batch, hidden_size*2)
        z_linear, r_linear = torch.chunk(2, 1) # seperate the results, (batch, hidden_size)
        zt = F.sigmoid(z_linear)
        rt = F.sigmoid(r_linear)
        h_hat = F.tanh(F.linear(xt, self.Wx) + F.linear(rt.mul(h_prevt), self.Wh)) # (batch, hidden_size)
        ht = (1 - zt).mul(h_prevt) + zt.mul(h_hat) # (batch, hidden_size)
        return ht
        
class GRU(nn.Module):
    '''
    没写完 numpy_layers 怎么实现?
    '''
    def __init__(self, input_size, hidden_size, num_of_layers):
        self.layers = []
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_of_layers = num_of_layers
        
    
    def forward(self, inputs, h0):
        '''
        Arguments:
        input -- (seq_len, batch, input_size)
        h0 -- (number_layers, batch, hidden_size)
        '''
        seq_len, batch, input_size = inputs.shape
        
        assert(input_size == self.input_size)
        
        ht = h0
        y = np.zeros((seq_len, batch, hidden_size))

gru = nn.GRU()