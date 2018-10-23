"""Implement model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.dilated_resnet import drn_d_107
from lib.utils import wrap_cuda


class CNN(nn.Module):
    """Multi-label Conv-Net classifier."""

    def __init__(self, num_classes):
        """Constructor."""
        super(CNN, self).__init__()
        self.drn = drn_d_107(num_classes=num_classes, pool_size=14)
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


class CNNRNN(nn.Module):
    """Implementation of CNN-RNN model.

    https://arxiv.org/pdf/1604.04573.pdf
    """

    def __init__(self, num_classes, n_feats_lstm):
        """Constructor."""
        super(CNNRNN, self).__init__()
        # Conv-net
        self.conv_net = CNN(num_classes=num_classes)
        self.n_feats_conv_net = self.conv_net.drn.get_n_features_out()

        # LSTM
        self.lstm = LSTMCell(n_feats_lstm, self.n_feats_conv_net)
        self.reset_lstm_hidden_states()

        # Label embedding matrix
        self.U_l = nn.Parameter(torch.rand(num_classes, n_feats_lstm))

        # Projection layers
        self.projection_I = nn.Parameter(torch.rand(self.n_feats_conv_net, n_feats_lstm))
        self.projection_O = nn.Parameter(torch.rand(self.n_feats_conv_net, n_feats_lstm))

    def reset_lstm_hidden_states(self):
        """Reset hidden state of lstm."""
        self.hidden_state = (wrap_cuda(torch.randn(1, self.n_feats_conv_net)),
                             wrap_cuda(torch.randn(1, self.n_feats_conv_net)))

    def forward(self, inputs, labels_one_hot):
        """Forward pass for training.

        Args:
            inputs (torch.Tensor): input images.
            labels_one_hot (list): list of lists with label one-hot vectors.

        Returns:
            predictions_batch (list): list of lists containing tensor with output
            class probabilities.

        """
        predictions_batch = []
        img_embeddings = self.conv_net.get_features(inputs).squeeze()
        batch_size = len(labels_one_hot)

        for idx, labels in enumerate(labels_one_hot):
            img_embedding = img_embeddings[idx].unsqueeze(0)
            n_labels = len(labels)
            self.reset_lstm_hidden_states()
            predictions = []

            # first prediction
            x_t = F.relu(torch.mm(img_embedding, self.projection_I))
            pred = torch.mm(x_t, self.U_l.transpose(0, 1))
            predictions.append(pred)

            for idy in range(n_labels - 1):
                label_embedding = torch.mm(labels[idy], self.U_l)
                lstm_output, c = self.lstm(label_embedding,
                                           self.hidden_state)
                self.hidden_state = (lstm_output, c)
                x_t = F.relu(torch.mm(img_embedding, self.projection_I) + \
                    torch.mm(lstm_output[0], self.projection_O))
                pred = torch.mm(x_t, self.U_l.transpose(0, 1))
                predictions.append(pred)
            predictions_batch.append(predictions)

        return predictions_batch

    def inference(self, img):
        """Forward pass for inference.

        Args:
            img (torch.Tensor): input image.

        """
        pass

        
def test_cnn_rnn_pipeline():
    """Test the different modules that goes into the CNN-RNN implementation."""
    num_classes, n_feats_lstm = 7178, 64
    batch_size = 1
    img_size = 128

    conv_net = CNN(num_classes=num_classes)
    n_feats_conv_net = conv_net.drn.get_n_features_out()
    lstm = LSTMCell(n_feats_lstm, n_feats_conv_net)
    U_l = nn.Parameter(torch.rand(num_classes, n_feats_lstm))
    projection_I = nn.Parameter(torch.rand(n_feats_conv_net, n_feats_lstm))
    projection_O = nn.Parameter(torch.rand(n_feats_conv_net, n_feats_lstm))

    labels_one_hot = torch.zeros(batch_size, num_classes)
    labels_one_hot[0][3] = 1
    input_conv_net = torch.ones(batch_size, 3, img_size, img_size)
    test_hidden = (torch.ones(1, n_feats_conv_net), torch.ones(1, n_feats_conv_net))

    img_embedding = conv_net.get_features(input_conv_net).squeeze().unsqueeze(0)
    label_embedding = torch.mm(labels_one_hot, U_l)
     # assume 1 step here, will be equal to number of labels
    lstm_output, c = lstm(label_embedding, test_hidden)
    lstm_output = lstm_output.squeeze().unsqueeze(0)

    x_t = F.relu(torch.mm(img_embedding, projection_I) + torch.mm(lstm_output, projection_O))

    pred = F.softmax(torch.mm(x_t, U_l.transpose(0, 1)), dim=1)

    assert pred.shape == (batch_size, num_classes)


test_cnn_rnn_pipeline()
