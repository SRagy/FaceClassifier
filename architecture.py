from torch.nn import Conv2d 
import torch.nn as nn
import torch
from math import log2, ceil
from residual_blocks import ClassicBottleneck, InvertedBottleneck, ConvNeXtBlock


# Aim to constrain number of parameters to 21m. Expects 3x224x224 inputs. 
# for 'classic' stem, we get 
# |params| = conv_weights + conv_bias + bn_weights = in_chan x out_chan * 49 + out_chan + 128
# If out_channels is 96 then we get 9600 parameters.
class Stem(nn.Module):
    """The stem represents the first part of the network, taking the input images
    and performing aggressive downsampling. Two types are implemented, 'classic and 'patchify'.
    
    Classic uses an initial 7x7 convolution then a 3x3 max pooling.
    This imitates the initial block of Resnet and GoogeLeNet amongst others.
    
    see e.g. https://d2l.ai/chapter_convolutional-modern/cnn-design.html

    Patchify comes from the convnext architecture and uses a 4x4 non-overlapping convolution,
    i.e. with step size 4.
    
    see https://arxiv.org/pdf/2201.03545.pdf
    

    """
    def __init__(self, out_channels = 80, stem_type = 'patchify'):

        super().__init__()
        if stem_type == 'patchify':
            self.net = nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size=4, stride=4, padding = 0),
            nn.ReLU6(),
            nn.LazyBatchNorm2d())

        elif stem_type == 'classic':
            self.net = nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size=7, stride=2, padding=3),
            nn.ReLU6(),
            nn.LazyBatchNorm2d(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        return self.net(x)

# Alternative possible block arrangement [88,176,352,704],[2,2,4,2]
class Body(nn.Module):
    """The main body of the classifier, modelled primarily after ConvNeXt-T
    with some experimental alterations, such as using inverse bottleneck blocks
    to step down the resolution between each major segment of the architecture.
    The ConvNeXt architecture simply uses a strided convolution.

    Additionally, has to be bit smaller than ConvNeXt-T to fit into the required
    21m parameter limit.
    """
    def __init__(self, 
                 channels = [80,160,320,640],
                 blocks = [2,2,6,2]
                 ):
        """init function for the body of the network

        Args:
            channels (list, optional): The number of input and output channels of each block group. 
            Inverse bottlneck blocks are used to step between these. Defaults to [80,160,320,640].
            blocks (list, optional): The number of blocks in each block group. Defaults to [2,2,6,2].
        """
        
        super().__init__()
        self.blocks=[]
        
        for i in range(len(blocks)):
            num_blocks = blocks[i]
            num_channels = channels[i]

            for j in range(num_blocks):
                block = ConvNeXtBlock(num_channels, 4*num_channels)
                self.blocks.append(block)

            self.blocks.append(nn.LazyBatchNorm2d())
            downsampling_block = InvertedBottleneck(num_channels, 2*num_channels, stride=2, use_residual=False)
            if i < len(blocks) - 1: # Downsample except on final layer
                self.blocks.append(downsampling_block)

        self.net = nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    """A small MLP classifier to act as the head of the image classifier
    """
    def __init__(self, input_channels: int = 512, 
                 pool_dim: int = 1,
                 num_classes: int = 7001,
                 double_layer: bool =True):
        """The classifier at the head of the CNN.

        Args:
            input_channels (int, optional): Input features. Defaults to 512.
            pool_dim (int, optional): Number of dimensions after pooling for avg pool. Defaults to 1.
            num_classes (int, optional): Number of class labels to discriminate. Defaults to 7001.
            double_layer (bool, optional): If true, uses two linear layers, else just one. Defaults to True.
        """
        super().__init__()
        avg_layer = nn.AdaptiveAvgPool2d(pool_dim)
        flatten_layer = nn.Flatten()
        in_dim = input_channels*pool_dim**2
        
        if double_layer is True:
            self.net = nn.Sequential(avg_layer, flatten_layer, 
                                     nn.Linear(in_dim, in_dim),
                                     nn.Linear(in_dim, num_classes))
        else:
            self.net = nn.Sequential(avg_layer, flatten_layer, nn.Linear(in_dim, num_classes))
            

    def forward(self,x):
        return self.net(x)

    

class FaceNN(nn.Module):
    """Classifier designed for facial recognition on small datasets.
    """
    def __init__(self,
                 channels = [80,160,320,640],
                 blocks = [2,2,6,2],
                 stem_type = 'patchify',
                 pool_dim = 1,
                 num_classes = 7001,
                 double_layer = True
):
        """_summary_

        Args:
            channels (list, optional): The number of input and output channels of each block group. 
            Inverse bottlneck blocks are used to step between these. Defaults to [80,160,320,640].
            blocks (list, optional): The number of blocks in each block group. Defaults to [2,2,6,2].
            stem_type (str, optional): ConvNeXT patchify or resnet classic start of network. Defaults to 'patchify'.
            pool_dim (int, optional): Pooling dimensions for avg pooling before classifier. Defaults to 1.
            num_classes (int, optional): Number of class labels to discriminate. Defaults to 7001.
            double_layer (bool, optional): Whether to make a 2-layler MLP instead of single linear layer classifier. 
            Defaults to True.
        """

        super().__init__()
        self.stem = Stem(channels[0], stem_type)
        self.body = Body(channels, blocks)
        self.head = Head(channels[-1], pool_dim, num_classes, double_layer)

    def forward(self,x):
        x = self.stem(x)
        x = self.body(x)
        x = self.head(x)
        return x
        