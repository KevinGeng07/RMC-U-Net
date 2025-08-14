import torch
import torch.nn as nn


### Both implementations are taken from "jhhuang96": https://github.com/jhhuang96/ConvLSTM-PyTorch
class CGRU_cell(nn.Module):
    def __init__(self, shape, input_channels, filter_size, num_features):
        super().__init__()
        self.shape = shape
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features, 2 * self.num_features, self.filter_size, 1, self.padding, bias=False),
            nn.GroupNorm(2 * self.num_features // 32, 2 * self.num_features))

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features, self.num_features, self.filter_size, 1, self.padding, bias=False),
            nn.GroupNorm(self.num_features // 32, self.num_features))


    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        if hidden_state is None:
            htprev = torch.zeros(inputs.size(1), self.num_features, self.shape[0], self.shape[1]).cuda()
        else:
            htprev = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(htprev.size(0), self.input_channels, self.shape[0], self.shape[1]).cuda()
            else:
                x = inputs[index, ...]

            combined_1 = torch.cat((x, htprev), 1)
            gates = self.conv1(combined_1)

            zgate, rgate = torch.split(gates, self.num_features, dim=1)
            z = torch.sigmoid(zgate)
            r = torch.sigmoid(rgate)

            combined_2 = torch.cat((x, r * htprev),1)
            ht = self.conv2(combined_2)
            ht = torch.tanh(ht)
            htnext = (1 - z) * htprev + z * ht
            output_inner.append(htnext)
            htprev = htnext

        return htnext, output_inner

    def init_weights(self):
          for m in self.modules():
            if isinstance(m, nn.Conv2d):
              nn.init.orthogonal_(m.weight)


class CLSTM_cell(nn.Module):
    """ConvLSTMCell
    """
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CLSTM_cell, self).__init__()

        self.shape = shape
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features, 4 * self.num_features, self.filter_size, 1, self.padding, bias=False),
            nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features))


    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        if hidden_state is None:
            hx = torch.zeros(inputs.size(1), self.num_features, self.shape[0], self.shape[1]).cuda()
            cx = torch.zeros(inputs.size(1), self.num_features, self.shape[0], self.shape[1]).cuda()
        else:
            hx, cx = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(hx.size(0), self.input_channels, self.shape[0], self.shape[1]).cuda()
            else:
                x = inputs[index, ...]

            combined = torch.cat((x, hx), 1)
            gates = self.conv(combined)

            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner.append(hy)
            hx = hy
            cx = cy
        return hy, output_inner

    def init_weights(self):
          for m in self.modules():
            if isinstance(m, nn.Conv2d):
              nn.init.orthogonal_(m.weight)
