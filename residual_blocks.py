from torch.nn import Conv2d 
import torch.nn as nn
import torch

# This file contains various convolutional blocks, inspired by, but not necessarily identical to
# residual blocks commonly found in the literature. One such variation, is the choice to apply
# batch norm layers after activation rather than before.
# There is not a strong rationale for these differences; some of it is updating things to 
# have a more modern flavour (e.g. adding batch norm to original ResNet blocks), some of it
# is more arbitrary. It would be worth doing some additional testing to see what's best.

class ClassicBottleneck(nn.Module):
    """A classic residual bottlneck block, as in ResNet. Consists of a 
    1. 1x1 convolution (i.e. linear mixture of channels), 
    2. followed by a convolution with larger kernel (typically 3x3), 
    3. then another 1x1 convolution.

    The initial convolution reduces the number of channels and the final
    one increases them. Hence the middle bit acts on the bottleneck.

    The input is added to the output before a final activaition.
    """
    def __init__(self, ext_channels, int_channels, kernel_size=3, activation=nn.ReLU6(), norm=True):
        """init for classic bottleneck, more-or-less as in Resnet

        Args:
            ext_channels (int): exterior channel count (i.e input and output)
            int_channels (int): interior channel count for the sandwiched convolution.
            kernel_size (int): kernel size for interior convolution.
            activation (nn.Module, optional): Activation function. Defaults to nn.ReLU6().
        """
        super().__init__()
        self._layers=[]
        self.activation = activation

        initial_conv = Conv2d(
            in_channels = ext_channels, 
            out_channels = int_channels, 
            kernel_size= 1)
        
        self._layers.append(initial_conv)
        self._layers.append(activation)
        if norm==True:
            self._layers.append(nn.LazyBatchNorm2d())
        
        middle_conv = Conv2d(
            in_channels=int_channels,
            out_channels=int_channels,
            kernel_size=kernel_size,
            padding='same'
        )

        self._layers.append(middle_conv)
        self._layers.append(activation)
        if norm==True:
            self._layers.append(nn.LazyBatchNorm2d())


        final_conv = Conv2d(
            in_channels=int_channels,
            out_channels=ext_channels,
            kernel_size=1
        )

        self._layers.append(final_conv)

        self._layers = nn.Sequential(*self._layers)

    def forward(self, x):
        return self.activation(self._layers(x) + x)

class InvertedBottleneck(nn.Module):
    """An inverted bottleneck block as in MobileNetV2 architecture.
    The structure is as follows:
    1. An initial 1x1 convolution expands the channel number - followed by norm and activation
    2. A depthwise-separable convolution (typically 3x3) does the heavy lifting - followed by norm and activation
    3. A final 1x1 convolution to collect the channels again - followed by norm but not activation

    """
    def __init__(self, in_channels, out_channels, kernel_size=3, expansion_factor=6, 
                 stride=1, activation=nn.ReLU6(), norm=True, use_residual=True):
        """Initialisation for inverse bottleneck block

        Args:
            in_channels (int): The number of channels in the input
            out_channels (int): The number of channels in the output
            kernel_size (int): The size of the square kernel
            expansion_factor (int, optional): The factor by which we increase the number of inner channels.
            stride (int, optional):  Defaults to 1.
            activation (_type_, optional): Defaults to nn.ReLU6().
            norm (bool, optional): Whether we normalise. Defaults to True.
            residual (bool, optional): Whether to add residual or not. Defaults to True.
        """
        super().__init__()
        self.activation = activation
        mid_channels = in_channels*expansion_factor
        self._layers = []
        self.use_residual = use_residual

        assert isinstance(expansion_factor, int), "please choose an integer value for the expansion factor"
        if use_residual==True:
            assert stride==1 and in_channels==out_channels, "to add residual we need a stride of 1 and in_channels = out_channels"

        initial_conv = Conv2d(
            in_channels = in_channels, 
            out_channels = mid_channels, 
            kernel_size=1)
        
        self._layers.append(initial_conv)
        self._layers.append(activation)
        # Consider placing norm layers before activation
        if norm == True:
            self._layers.append(nn.LazyBatchNorm2d())

        total_padding = kernel_size-1
        padding_left = (kernel_size-1)//2
        padding_right = total_padding - padding_left

        interior_conv = Conv2d(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=[padding_left,padding_right],
            groups=mid_channels
        )

        self._layers.append(interior_conv)
        self._layers.append(activation)

        if norm == True:
            self._layers.append(nn.LazyBatchNorm2d())

        final_conv = Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1
        )

        self._layers.append(final_conv)
        if norm == True:
            self._layers.append(nn.LazyBatchNorm2d())

        self._layers = nn.Sequential(*self._layers)

    def forward(self, x):
        if self.use_residual:
            return self._layers(x) + x
        else:
            return self._layers(x)


class ConvNeXtBlock(nn.Module):
    """A ConvNeXt block, consists of a 
    1. 7x7 depthwise convolution, 
    2. followed by 1x1 convolution which expands the number of channels,
    3. then another 1x1 convolution which reduces them again.

    My implementation uses batchnorm instead of layernorm, but not for
    any especially good reason.

    The input is added to the output *without* a final activation.

    https://arxiv.org/pdf/2201.03545.pdf
    """
    def __init__(self, ext_channels, int_channels, kernel_size=7, activation=nn.GELU(), norm=True):
        """init for classic bottleneck, more-or-less as in Resnet

        Args:
            ext_channels (int): exterior channel count (i.e input and output)
            int_channels (int): interior channel count for the sandwiched convolution.
            kernel_size (int): kernel size for interior convolution.
            activation (nn.Module, optional): Activation function. Defaults to nn.ReLU6().
        """
        super().__init__()
        self._layers=[]
        self.activation = activation


        total_padding = kernel_size-1
        padding_left = (kernel_size-1)//2
        padding_right = total_padding - padding_left


        initial_conv = Conv2d(
            in_channels=ext_channels,
            out_channels=ext_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=[padding_left,padding_right],
            groups=ext_channels
        )

        self._layers.append(initial_conv)
        if norm==True:
            self._layers.append(nn.LazyBatchNorm2d())
        
        middle_conv = Conv2d(
            in_channels=ext_channels,
            out_channels=int_channels,
            kernel_size=1,
        )

        self._layers.append(middle_conv)
        self._layers.append(activation)

        final_conv = Conv2d(
            in_channels=int_channels,
            out_channels=ext_channels,
            kernel_size=1
        )

        self._layers.append(final_conv)

        self._layers = nn.Sequential(*self._layers)

    def forward(self, x):
        return self._layers(x) + x
