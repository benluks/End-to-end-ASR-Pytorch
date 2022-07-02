from math import floor, sqrt
import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.util import binarize, qlstm_cell


class VGGExtractor(nn.Module):
    ''' VGG extractor for ASR described in https://arxiv.org/pdf/1706.02737.pdf'''

    def __init__(self, input_dim):
        super(VGGExtractor, self).__init__()
        self.init_dim = 64
        self.hide_dim = 128
        in_channel, freq_dim, out_dim = self.check_dim(input_dim)
        self.in_channel = in_channel
        self.freq_dim = freq_dim
        self.out_dim = out_dim

        self.extractor = nn.Sequential(
            nn.Conv2d(in_channel, self.init_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.init_dim, self.init_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # Half-time dimension
            nn.Conv2d(self.init_dim, self.hide_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hide_dim, self.hide_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # Half-time dimension
        )

    def check_dim(self, input_dim):
        # Check input dimension, delta feature should be stack over channel.
        if input_dim % 13 == 0:
            # MFCC feature
            return int(input_dim/13), 13, (13//4)*self.hide_dim
        elif input_dim % 40 == 0:
            # Fbank feature
            return int(input_dim/40), 40, (40//4)*self.hide_dim
        else:
            raise ValueError(
            'Acoustic feature dimension for VGG should be 13/26/39(MFCC) or 40/80/120(Fbank) but got '+input_dim)

    def view_input(self, feature, feat_len):
        # downsample time
        feat_len = feat_len//4
        # crop sequence s.t. t%4==0
        if feature.shape[1] % 4 != 0:
            feature = feature[:, :-(feature.shape[1] % 4), :].contiguous()
        bs, ts, ds = feature.shape
        # stack feature according to result of check_dim
        feature = feature.view(bs, ts, self.in_channel, self.freq_dim)
        feature = feature.transpose(1, 2)

        return feature, feat_len

    def forward(self, feature, feat_len):
        # Feature shape BSxTxD -> BS x CH(num of delta) x T x D(acoustic feature dim)
        feature, feat_len = self.view_input(feature, feat_len)
        # Foward
        feature = self.extractor(feature)
        # BSx128xT/4xD/4 -> BSxT/4x128xD/4
        feature = feature.transpose(1, 2)
        #  BS x T/4 x 128 x D/4 -> BS x T/4 x 32D
        feature = feature.contiguous().view(feature.shape[0], feature.shape[1], self.out_dim)
        return feature, feat_len

class CNNExtractor(nn.Module):
    ''' A simple 2-layer CNN extractor for acoustic feature down-sampling'''

    def __init__(self, input_dim, out_dim):
        super(CNNExtractor, self).__init__()

        self.out_dim = out_dim
        self.extractor = nn.Sequential(
            nn.Conv1d(input_dim, out_dim, 4, stride=2, padding=1),
            nn.Conv1d(out_dim, out_dim, 4, stride=2, padding=1),
        )

    def forward(self, feature, feat_len):
        # Fixed down-sample ratio
        feat_len = feat_len//4
        # Channel first
        feature = feature.transpose(1,2) 
        # Foward
        feature = self.extractor(feature)
        # Channel last
        feature = feature.transpose(1, 2)

        return feature, feat_len


class RNNLayer(nn.Module):
    ''' RNN wrapper, includes time-downsampling'''

    def __init__(self, input_dim, module, dim, bidirection, dropout, layer_norm, sample_rate, sample_style, proj, device=None):
        super(RNNLayer, self).__init__()
        # Setup
        rnn_out_dim = 2*dim if bidirection else dim
        self.out_dim = sample_rate * \
            rnn_out_dim if sample_rate > 1 and sample_style == 'concat' else rnn_out_dim
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.sample_rate = sample_rate
        self.sample_style = sample_style
        self.proj = proj

        if self.sample_style not in ['drop', 'concat']:
            raise ValueError('Unsupported Sample Style: '+self.sample_style)

        # Recurrent layer
        else:
            if module == 'QLSTM':
                self.layer = QLSTM(input_size=input_dim, hidden_size=dim, num_layers=1, 
                                    batch_first=True, bias=False, bidirectional=bidirection)
            else:
                self.layer = getattr(nn, module.upper())(
                    input_dim, dim, bidirectional=bidirection, num_layers=1, batch_first=True)

        # Regularizations
        if self.layer_norm:
            self.ln = nn.LayerNorm(rnn_out_dim)
        if self.dropout > 0:
            self.dp = nn.Dropout(p=dropout)

        # Additional projection layer
        if self.proj:
            self.pj = nn.Linear(rnn_out_dim, rnn_out_dim)

    def forward(self, input_x, x_len):
        # Forward RNN
        if not self.training:
            self.layer.flatten_parameters()
        # ToDo: check time efficiency of pack/pad
        #input_x = pack_padded_sequence(input_x, x_len, batch_first=True, enforce_sorted=False)
        output, _ = self.layer(input_x)
        #output,x_len = pad_packed_sequence(output,batch_first=True)

        # Normalizations
        if self.layer_norm:
            output = self.ln(output)
        if self.dropout > 0:
            output = self.dp(output)

        # Perform Downsampling
        if self.sample_rate > 1:
            batch_size, timestep, feature_dim = output.shape
            x_len = x_len//self.sample_rate

            if self.sample_style == 'drop':
                # Drop the unselected timesteps
                output = output[:, ::self.sample_rate, :].contiguous()
            else:
                # Drop the redundant frames and concat the rest according to sample rate
                if timestep % self.sample_rate != 0:
                    output = output[:, :-(timestep % self.sample_rate), :]
                output = output.contiguous().view(batch_size, int(
                    timestep/self.sample_rate), feature_dim*self.sample_rate)

        if self.proj:
            output = torch.tanh(self.pj(output))

        return output, x_len


class BaseAttention(nn.Module):
    ''' Base module for attentions '''

    def __init__(self, temperature, num_head):
        super().__init__()
        self.temperature = temperature
        self.num_head = num_head
        self.softmax = nn.Softmax(dim=-1)
        self.reset_mem()

    def reset_mem(self):
        # Reset mask
        self.mask = None
        self.k_len = None

    def set_mem(self, prev_att):
        pass

    def compute_mask(self, k, k_len):
        # Make the mask for padded states
        self.k_len = k_len
        bs, ts, _ = k.shape
        self.mask = np.zeros((bs, self.num_head, ts))
        for idx, sl in enumerate(k_len):
            self.mask[idx, :, sl:] = 1  # ToDo: more elegant way?
        self.mask = torch.from_numpy(self.mask).to(
            k_len.device, dtype=torch.bool).view(-1, ts)  # BNxT

    def _attend(self, energy, value):
        attn = energy / self.temperature
        attn = attn.masked_fill(self.mask, -np.inf)
        attn = self.softmax(attn)  # BNxT
        output = torch.bmm(attn.unsqueeze(1), value).squeeze(
            1)  # BNxT x BNxTxD-> BNxD
        return output, attn


class ScaleDotAttention(BaseAttention):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, num_head):
        super().__init__(temperature, num_head)

    def forward(self, q, k, v):
        ts = k.shape[1]
        energy = torch.bmm(q.unsqueeze(1), k.transpose(
            1, 2)).squeeze(1)  # BNxD * BNxDxT = BNxT
        output, attn = self._attend(energy, v)

        attn = attn.view(-1, self.num_head, ts)  # BNxT -> BxNxT

        return output, attn


class LocationAwareAttention(BaseAttention):
    ''' Location-Awared Attention '''

    def __init__(self, kernel_size, kernel_num, dim, num_head, temperature):
        super().__init__(temperature, num_head)
        self.prev_att = None
        self.loc_conv = nn.Conv1d(
            num_head, kernel_num, kernel_size=2*kernel_size+1, padding=kernel_size, bias=False)
        self.loc_proj = nn.Linear(kernel_num, dim, bias=False)
        self.gen_energy = nn.Linear(dim, 1)
        self.dim = dim

    def reset_mem(self):
        super().reset_mem()
        self.prev_att = None

    def set_mem(self, prev_att):
        self.prev_att = prev_att

    def forward(self, q, k, v):
        bs_nh, ts, _ = k.shape
        bs = bs_nh//self.num_head

        # Uniformly init prev_att
        if self.prev_att is None:
            self.prev_att = torch.zeros((bs, self.num_head, ts)).to(k.device)
            for idx, sl in enumerate(self.k_len):
                self.prev_att[idx, :, :sl] = 1.0/sl

        # Calculate location context
        loc_context = torch.tanh(self.loc_proj(self.loc_conv(
            self.prev_att).transpose(1, 2)))  # BxNxT->BxTxD
        loc_context = loc_context.unsqueeze(1).repeat(
            1, self.num_head, 1, 1).view(-1, ts, self.dim)   # BxNxTxD -> BNxTxD
        q = q.unsqueeze(1)  # BNx1xD

        # Compute energy and context
        energy = self.gen_energy(torch.tanh(
            k+q+loc_context)).squeeze(2)  # BNxTxD -> BNxT
        output, attn = self._attend(energy, v)
        attn = attn.view(bs, self.num_head, ts)  # BNxT -> BxNxT
        self.prev_att = attn

        return output, attn


class QLSTM(nn.LSTM):

    def __init__(self, *args, quant='bin', **kwargs):

        super().__init__(**kwargs)
        self.device = kwargs['device'] if 'device' in kwargs.keys() else torch.device('cpu')
        self.init_constant = kwargs['init_constant'] if 'init_constant' in kwargs.keys() else 6.
        self.quant = quant

        if self.quant:
            # layer-specific initializations 
            for layer in range(self.num_layers):  
                
                # add batchnorms
                bn_gates = nn.BatchNorm1d(8)
                bn_c = nn.BatchNorm1d(1)
                
                bn_gates.bias.requires_grad_(False)
                bn_c.bias.requires_grad_(False)
                
                self.add_module(f'bn_l{layer}', bn_gates)
                self.add_module(f'bn_c_l{layer}', bn_c)

                # add scaling factor W0
                l_input_size = self.input_size if layer == 0 else self.hidden_size
                W0_ih = sqrt(self.init_constant / (l_input_size + 4 * self.hidden_size)) / 2
                W0_hh = sqrt(self.init_constant / (self.hidden_size + 4 * self.hidden_size)) / 2

                setattr(self, f'W0_ih_l{layer}', W0_ih)
                setattr(self, f'W0_hh_l{layer}', W0_hh)

                if self.bidirectional:
                    # add batchnorms
                    bn_gates_reverse = nn.BatchNorm1d(8)
                    bn_c_reverse = nn.BatchNorm1d(1)
                    
                    bn_gates_reverse.bias.requires_grad_(False)
                    bn_c_reverse.bias.requires_grad_(False)
                    
                    self.add_module(f'bn_l{layer}_reverse', bn_gates_reverse)
                    self.add_module(f'bn_c_l{layer}_reverse', bn_c_reverse)


    def _get_layer_params(self, layer, reverse=False):
        """
        Get the appropriate parameters for a given layer during a forward pass
        """
        rev = "_reverse" if reverse else ""
        tail_idx = 2 + len(rev)
        
        layer_params = [p for n, p in self.named_parameters() if n[-tail_idx:] == f"l{layer}{rev}"]
        if not self.bias: layer_params += [0, 0]
        layer_params.append(self.device)
        if self.quant:
            layer_params.append(getattr(self, f'bn_l{layer}{rev}'))
            layer_params.append(getattr(self, f'bn_c_l{layer}{rev}'))
        
        return layer_params


    def binarize(self, par, name, device):
        """
        placeholder to signal which W0 values to pass
        """
        _, place, layer = name.split("_")[:3]
        W0 = getattr(self, f"W0_{place}_{layer}")
        return binarize(par, W0, device=device)


    def forward(self, input, h_0=None):

        T = input.size(0) if not self.batch_first else input.size(1)
        B = input.size(1) if not self.batch_first else input.size(0)
        
        # final hidden states (h and c) for each layer
        h_t = []

        for layer in range(self.num_layers):
        
            layer_params = self._get_layer_params(layer)
            outputs = []

            if self.bidirectional:
                layer_params_reverse = self._get_layer_params(layer, reverse=True)
                outputs_reverse = []
                
                # hidden states if given h_0 if bidirectional
                if h_0:
                    hidden = (h_0[0][2*layer], h_0[1][2*layer])
                    hidden_reverse = (h_0[0][2*layer+1], h_0[1][2*layer+1])
                else:
                    hidden = 2*(torch.zeros(B, self.hidden_size, device=self.device),)
                    hidden_reverse = 2*(torch.zeros(B, self.hidden_size, device=self.device),)
        
        # init hidden states if not bidirectional
        else:
            if h_0:
                hidden = (h_0[0][layer], h_0[1][layer]) if self.num_layers > 1 else h_0
            else:
                hidden = 2*(torch.zeros(B, self.hidden_size, device=self.device),)

        # loop through time steps
        for t in range(T):  
            # TODO: implement successive layer forward pass for bidirectionality
            input_t = input[:, t, :] if self.batch_first else input[t]
            hidden = qlstm_cell(input_t, hidden, *layer_params)
            outputs.append(hidden[0])

            if self.bidirectional:
                input_t_reverse = input[:, -(t+1), :] if self.batch_first else input[-(t+1)]
                hidden_reverse = qlstm_cell(input_t_reverse, hidden_reverse, *layer_params_reverse)
                outputs_reverse = [hidden_reverse[0]] + outputs_reverse
        
        # all time-steps are done, end T loop
        # -----------------------------------
        h_t.append(hidden)
        outputs = torch.stack(outputs, 1 if self.batch_first else 0)

        # reverse outputs
        if self.bidirectional:
            h_t.append(hidden_reverse)
            # outputs_reverse is shape [B, T, H], we want input to be [B, T, 2*H]
            outputs_reverse = torch.stack(outputs_reverse, 1 if self.batch_first else 0)
            outputs = torch.cat((outputs, outputs_reverse), dim=-1)
            
        # prev hidden states as following layer's input      
        input = outputs
        
        # h_t is [(h, c), (h, c), ...], we want to separate into lists [[h_0, h_1, ...], [c_0, c_1, ...]]
        h_t, c_t = list(zip(*h_t))
        h_t, c_t = torch.stack(h_t, 0), torch.stack(c_t, 0)

        return outputs, (h_t, c_t)