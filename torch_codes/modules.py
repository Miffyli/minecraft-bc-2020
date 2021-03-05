#
# PyTorch networks and modules
#

from collections import OrderedDict
import math

import torch
from torch import nn
import torch.nn.functional as F

# References:
# [1] IMPALA. https://arxiv.org/pdf/1802.01561.pdf
# [2] R2D3. https://arxiv.org/pdf/1909.01387.pdf
# [3] Unixpickle's work https://github.com/amiranas/minerl_imitation_learning/blob/master/model.py#L104

class ResidualBlock(nn.Module):
    """
    Residual block from R2D3/IMPALA

    Taken from [1,2]
    """

    def __init__(self, num_channels, first_conv_weight_scale):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)

        # Copy paste from [3]
        self.bias1 = nn.Parameter(torch.zeros([num_channels, 1, 1]))
        self.bias2 = nn.Parameter(torch.zeros([num_channels, 1, 1]))
        self.bias3 = nn.Parameter(torch.zeros([num_channels, 1, 1]))
        self.bias4 = nn.Parameter(torch.zeros([num_channels, 1, 1]))
        self.scale = nn.Parameter(torch.ones([num_channels, 1, 1]))

        # FixUp init (part of it):
        #  - Final Convs in residual branches initialized
        #    to zero
        #  - Other convs in residual branches initialized
        #    to a scaled value
        #  - Biases handled manually as in [3]
        with torch.no_grad():
            self.conv2.weight *= 0
            self.conv1.weight *= first_conv_weight_scale

    def forward(self, x):
        x = F.relu(x, inplace=True)
        original = x

        # Copy/paste from [3]
        x = x + self.bias1
        x = self.conv1(x)
        x = x + self.bias2

        x = F.relu(x, inplace=True)

        x = x + self.bias3
        x = self.conv2(x)
        x = x * self.scale
        x = x + self.bias4

        return original + x


class ResNetHead(nn.Module):
    """
    A small residual network CNN head for processing images.

    Architecture is from IMPALA paper in Fig 3 [1]
    """

    def __init__(self, in_channels=3, filter_sizes=(16, 32, 32), add_extra_block=False):
        super().__init__()
        self.num_total_blocks = len(filter_sizes) + int(add_extra_block)
        self.blocks = []

        # Scaler for FixUp mid-most convolutions.
        # Scaling is L^(-1/(2m - 2)) . In our case m = 2 (two layers in branch),
        # so our rescaling is L^(-1/2) = 1 / sqrt(L).
        # L is number of residual branches in our network.
        # Each block in IMPALA has two branches.
        first_conv_weight_scale = 1 / (math.sqrt(self.num_total_blocks * 2))
        input_channels = in_channels
        for output_channels in filter_sizes:
            block = [
                nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2),
                ResidualBlock(output_channels, first_conv_weight_scale),
                ResidualBlock(output_channels, first_conv_weight_scale),
            ]
            self.blocks.extend(block)
            input_channels = output_channels
        # Number of blocks without max pooling
        if add_extra_block:
            self.blocks.extend((
                ResidualBlock(output_channels, first_conv_weight_scale),
                ResidualBlock(output_channels, first_conv_weight_scale)
            ))
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        x = self.blocks(x)
        # Flatten
        x = x.reshape(x.shape[0], -1)
        x = F.relu(x, inplace=True)
        return x


# Some specific ResNet sizes
# for different resolutions to avoid too
# squuezed or large feaeture sizes.
def ResNetHeadFor64x64(in_channels):
    return ResNetHead(in_channels=in_channels)


def ResNetHeadFor32x32(in_channels):
    return ResNetHead(in_channels=in_channels, filter_sizes=(16, 32))


def ResNetHeadFor64x64DoubleFilters(in_channels):
    # As in [3]
    return ResNetHead(in_channels=in_channels, filter_sizes=(32, 64, 64))


def ResNetHeadFor64x64DoubleFiltersWithExtra(in_channels):
    # As in [3]
    return ResNetHead(in_channels=in_channels, filter_sizes=(32, 64, 64), add_extra_block=True)


def ResNetHeadFor64x64QuadrupleFilters(in_channels):
    return ResNetHead(in_channels=in_channels, filter_sizes=(64, 128, 128))


class NatureDQNHead(nn.Module):
    """The CNN head from Nature DQN paper"""

    def __init__(self, in_channels=3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.head(x)
        x = x.reshape(x.shape[0], -1)
        return x


class IMPALANetwork(nn.Module):
    """
    IMPALA network [1] but without recurrence and
    with bit different network sizes.
    I.e. it takes in an image and some 1D features
    ("additional features"). Image is processed
    through a small ResNet network, and other
    features are appended to this output
    later in the network.

    Output is a dictionary with same keys
    as output_dict, all outputs being linear
    activations
    """

    def __init__(
        self,
        image_shape,
        output_dict,
        num_additional_features,
        cnn_head_class="ResNetHead",
        latent_size=512,
        num_heads=None
    ):
        """
        Parameters:
            image_shape (List): Shape of the input images in CxHxW
            output_dict (Dict of str: int): Names and dimensions of outputs
                to produce
            num_additional_features (int): Number of additional
                features appended later
            cnn_head_class (str): Name of the class (visible to here)
                to use as preprocess for images
            latent_size (int): Size of the latent vector after concatenating
                image and additional observations
            num_heads (int): Number of different output heads. None is converted
                to one.
        """
        super().__init__()
        self.num_heads = num_heads if num_heads is not None else 1
        self.output_dict = OrderedDict(output_dict)
        self.total_output_size = sum(list(self.output_dict.values()))
        self.num_additional_features = num_additional_features
        # SuperSecure^{tm}
        cnn_head_class = eval(cnn_head_class)
        self.cnn_head = cnn_head_class(
            in_channels=image_shape[0],
        )

        # Run something through the network to get the shape
        self.cnn_feat_size = self.cnn_head(torch.rand(*((1,) + image_shape))).shape[1]

        # Layer for additional features
        self.feats_fc1 = nn.Linear(self.num_additional_features, 128)

        # Append additional features
        self.num_combined_features = self.cnn_feat_size + 128

        # Create outputs. All outputs are one big list.
        self.final_fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.num_combined_features, latent_size),
                nn.ReLU(),
                nn.Linear(latent_size, self.total_output_size)
            )
            for i in range(self.num_heads)
        ])

    def forward(self, image_observation, additional_features, head_indeces=None):
        # Normalize image (uint8)
        x = image_observation.float() / 255.0

        x = self.cnn_head(x)

        # Process additional features through one layer
        additional_features = self.feats_fc1(additional_features)
        additional_features = F.relu(additional_features)

        x = torch.cat((x, additional_features), dim=1)

        if head_indeces is None:
            x = self.final_fcs[0](x)
        else:
            # Run different batch elements through different heads.
            # TODO this should probably be parallelized somehow...
            # TODO normalize gradients?
            out_x = torch.zeros(x.shape[0], self.total_output_size).to(x.device)
            for batch_i in range(x.shape[0]):
                out_x[batch_i] = self.final_fcs[head_indeces[batch_i]](x[batch_i])
            x = out_x

        # Split to different dicts according to output sizes
        outputs = {}
        i = 0
        for name, size in self.output_dict.items():
            outputs[name] = x[:, i:i + size]
            i += size

        return outputs


class IMPALANetworkWithLSTM(nn.Module):
    """
    IMPALA network [1].
    I.e. it takes in an image and some 1D features
    ("additional features"). Image is processed
    through a small ResNet network, and other
    features are appended to this output
    later in the network.

    Output is a dictionary with same keys
    as output_dict, all outputs being linear
    activations.
    """

    def __init__(
        self,
        image_shape,
        output_dict,
        num_additional_features,
        cnn_head_class="ResNetHead",
        latent_size=512,
        num_heads=None
    ):
        """
        Parameters:
            image_shape (List): Shape of the input images in CxHxW
            output_dict (Dict of str: int): Names and dimensions of outputs
                to produce
            num_additional_features (int): Number of additional
                features appended later
            cnn_head_class (str): Name of the class (visible to here)
                to use as preprocess for images
            latent_size (int): Number of units for the hidden LSTM state
        """
        assert num_heads is None, "Sub-tasks not supported for LSTM version yet"
        super().__init__()
        self.latent_size = latent_size
        self.output_dict = OrderedDict(output_dict)
        self.total_output_size = sum(list(self.output_dict.values()))
        self.num_additional_features = num_additional_features
        # SuperSecure^{tm}
        cnn_head_class = eval(cnn_head_class)
        self.cnn_head = cnn_head_class(
            in_channels=image_shape[0],
        )

        # Run something through the network to get the shape
        self.cnn_feat_size = self.cnn_head(torch.rand(*((1,) + image_shape))).shape[1]
        self.cnn_fc = nn.Linear(self.cnn_feat_size, 128)

        # Append additional features
        self.num_combined_features = 128 + self.num_additional_features

        self.lstm = nn.LSTM(
            input_size=self.num_combined_features,
            hidden_size=self.latent_size,
            num_layers=1
        )

        # Do all outputs as one big list
        self.final_fc = nn.Linear(self.latent_size, self.total_output_size)

    def get_initial_state(self, batch_size):
        """
        Return initial hidden state (at the start of the episode),
        i.e. zero vectors.
        """
        device = self.lstm.weight_hh_l0.device
        h = torch.zeros(1, batch_size, self.latent_size).to(device)
        c = torch.zeros(1, batch_size, self.latent_size).to(device)
        return h, c

    def forward(self, image_observation, additional_features, hidden_states=None, head_indeces=None, return_sequence=False):
        """
        Trajectory/timesteps first. Returns hidden states (h, c).
        """
        assert hidden_states is not None, "No hidden states provided"
        assert head_indeces is None, "Sub-tasks not supported for LSTM yet"

        # Normalize image (uint8)
        x = image_observation.float() / 255.0

        # Flatten batch and timestep axis into one to run through
        # CNN head.
        # Tested to work as expected with tests/test_time_parallelization.py
        x_shape = x.shape
        x = x.reshape(x_shape[0] * x_shape[1], x_shape[2], x_shape[3], x_shape[4])
        x = self.cnn_head(x)
        # Fully-connected
        x = F.relu(self.cnn_fc(x), inplace=True)
        # Now return to original seq_len, batch_size dim
        x = x.view(x_shape[0], x_shape[1], -1)

        # Add additional features
        x = torch.cat((x, additional_features), dim=2)

        # Run through lstms
        x, hidden_states = self.lstm(x, hidden_states)

        if not return_sequence:
            # Restrict to only last element in sequence
            x = x[-1]

        # Aaaand final mapping to output
        x = self.final_fc(x)

        # Split to different dicts according to output sizes
        outputs = {}
        i = 0
        for name, size in self.output_dict.items():
            if return_sequence:
                # Go over sequence length as well
                outputs[name] = x[:, :, i:i + size]
            else:
                outputs[name] = x[:, i:i + size]
            i += size

        return outputs, hidden_states
