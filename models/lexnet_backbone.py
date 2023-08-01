import torch
from torch import nn


def conv3x3(in_channels, out_channels, stride=1):
    """
    Convolution 3x3

    Parameters
    ----------
    in_channels: integer
        Number of input channels

    out_channels: integer
        Number of output channels

    stride: integer
        Stride of the convolution

    Returns
    -------
    feature_maps: array
        Feature maps from the convolution
    """
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


def linear3x3(in_channels, out_channels, stride=1):
    """
    Convolution 3x3 per channel

    Parameters
    ----------
    in_channels: integer
        Number of input channels

    out_channels: integer
        Number of output channels

    stride: integer
        Stride of the convolution

    Returns
    -------
    feature_maps: array
        Feature maps from the convolution
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=in_channels,
        bias=False,
    )


class LEResidualBlock(nn.Module):
    """Class used to create a LERes block"""

    # class attribute
    NUM_LAYERS = 2

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        linear_transformation=False,
        baseline_activation_function="relu",
    ):
        """
        Parameters
        ----------
        in_channels: integer
            Number of input channels

        out_channels: integer
            Number of output channels

        stride: integer
            Stride of the convolution

        linear_transformation: boolean
            True if out_channels > in_channels

        baseline_activation_function: string
            Activation function of the last layer
        """
        super(LEResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, in_channels)
        self.linear1 = linear3x3(in_channels, in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.linear_transformation = linear_transformation
        if baseline_activation_function == "sigmoid":
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.linear_transformation:
            residual = torch.cat([residual, out], dim=1)
            x1 = self.linear1(out)
            out = torch.cat([out, x1], dim=1)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.last_activation(out)
        return out

    def block_conv_info(self):
        block_kernel_sizes = [3, 3]
        block_strides = [1, 1]
        block_paddings = [1, 1]
        return block_kernel_sizes, block_strides, block_paddings


class LEXNet_features(nn.Module):
    """Class used to create the CNN backbone of LEXNet"""

    def __init__(
        self,
        block=LEResidualBlock,
        lyrs=[2, 2],
        ini=True,
        baseline_activation_function="relu",
    ):
        """
        Parameters
        ----------
        block: string
            Name of the block used to build the network

        lyrs: array
            Number of layers per block

        ini: boolean
            Initialize weights of the convolutional layers if True

        baseline_activation_function: string
            Activation function of the last layer
        """
        super(LEXNet_features, self).__init__()
        self.kernel_sizes = []
        self.strides = []
        self.paddings = []
        self.lyrs = lyrs
        self.in_channels = 8
        self.conv = conv3x3(1, 8)
        self.bn = nn.BatchNorm2d(8)
        self.kernel_sizes.append(3)
        self.strides.append(1)
        self.paddings.append(1)
        self.layer1 = self.make_layer(block, 16, lyrs[0], 1)
        self.layer2 = self.make_layer(
            block, 32, lyrs[1], 1, baseline_activation_function
        )
        if ini:
            self.initialize_weights()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.layer1(out)
        out = self.layer2(out)
        return out

    def make_layer(
        self, block, out_channels, blocks, stride=1, baseline_activation_function="relu"
    ):
        linear_transformation = False
        if out_channels > self.in_channels:
            linear_transformation = True
        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, linear_transformation)
        )
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(
                block(
                    out_channels,
                    out_channels,
                    stride,
                    baseline_activation_function=baseline_activation_function,
                )
            )
        for each_block in layers:
            (
                block_kernel_sizes,
                block_strides,
                block_paddings,
            ) = each_block.block_conv_info()
            self.kernel_sizes.extend(block_kernel_sizes)
            self.strides.extend(block_strides)
            self.paddings.extend(block_paddings)
        return nn.Sequential(*layers)

    def initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings

    def num_layers(self):
        return (
            self.block.NUM_LAYERS * self.lyrs[0]
            + self.block.NUM_LAYERS * self.lyrs[2]
            + 1
        )

    def __repr__(self):
        template = "BASE{}"
        return template.format(self.num_layers() + 1)


def lexnet_backbone(baseline_activation_function="relu"):
    """
    Build the CNN backbone of LEXNet

    Parameters
    ----------
    baseline_activation_function: string
        Activation function of the last layer

    Returns
    -------
    model: array
        CNN backbone of LEXNet
    """
    model = LEXNet_features(baseline_activation_function=baseline_activation_function)
    return model
