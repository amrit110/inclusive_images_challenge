"""Implement model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.dilated_resnet import drn_d_107


class CNN(nn.Module):
    """Multi-label Conv-Net classifier."""

    def __init__(self, num_classes=1000):
        """Constructor."""
        super(CNN, self).__init__()
        self.drn = drn_d_107(num_classes=num_classes, pool_size=28)
        self.num_classes = num_classes

    def forward(self, x):
        """Forward pass."""
        out = self.drn(x)
        output = torch.sigmoid(out[0])
        return output

    def get_features(self, x):
        """Forward pass and get image embedding."""
        out = self.drn(x)
        return out[1]


class LSTMCell(nn.Module):
    """Custom LSTM implementation since we want ReLU activations."""

    def __init__(self, input_size, hidden_size, nlayers=1, dropout=0):
        """Constructor."""
        super(LSTMCell, self).__init__()

        self.nlayers = nlayers
        self.dropout = nn.Dropout(p=dropout)

        ih, hh = [], []
        for i in range(nlayers):
            ih.append(nn.Linear(input_size, 4 * hidden_size))
            hh.append(nn.Linear(hidden_size, 4 * hidden_size))

        self.w_ih = nn.ModuleList(ih)
        self.w_hh = nn.ModuleList(hh)

    def forward(self, input, hidden):
        """Define the forward computation of the LSTMCell."""
        hy, cy = [], []

        for i in range(self.nlayers):
            hx, cx = hidden[0][i], hidden[1][i]
            gates = self.w_ih[i](input) + self.w_hh[i](hx)
            i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 1)

            i_gate = F.relu(i_gate)
            f_gate = F.relu(f_gate)
            c_gate = F.relu(c_gate)
            o_gate = F.relu(o_gate)

            ncx = (f_gate * cx) + (i_gate * c_gate)
            nhx = o_gate * F.relu(ncx)
            cy.append(ncx)
            hy.append(nhx)
            input = self.dropout(nhx)

        hy, cy = torch.stack(hy, 0), torch.stack(cy, 0)
        return hy, cy


def test_cnn_rnn_pipeline():
    """Test the different modules that goes into the CNN-RNN implementation."""
    num_classes, n_feats_lstm, n_feats_conv_net = 7178, 64, 512
    batch_size = 2
    img_size = 224

    conv_net = CNN(num_classes=num_classes)
    lstm = LSTMCell(n_feats_lstm, n_feats_conv_net)

    U_l = nn.Parameter(torch.rand(num_classes, n_feats_lstm))
    projection_I = nn.Parameter(torch.rand(n_feats_conv_net, n_feats_lstm))
    projection_O = nn.Parameter(torch.rand(n_feats_conv_net, n_feats_lstm))

    labels_one_hot = torch.zeros(batch_size, num_classes)
    labels_one_hot[0][3] = 1
    input_conv_net = torch.zeros(batch_size, 3, img_size, img_size)
    test_hidden = (torch.randn(1, 1, n_feats_conv_net), torch.randn(1, 1, n_feats_conv_net))

    img_embedding = conv_net.get_features(input_conv_net).squeeze()
    label_embedding = torch.mm(labels_one_hot, U_l)
     # assume 1 step here, will be equal to number of labels
    lstm_output = lstm(label_embedding, test_hidden)[0].squeeze()

    x_t = F.relu(torch.mm(img_embedding, projection_I) + torch.mm(lstm_output, projection_O))

    pred = F.softmax(torch.mm(x_t, U_l.transpose(0, 1)), dim=1)

    assert pred.shape == (batch_size, num_classes)


test_cnn_rnn_pipeline()
