import math
import torch.nn as nn
import torch

__all__ = ['resnet34','resnet50','resnet101']


# 首先定义bottleneck块的1x1 和3x3 卷积

def conv1x1(in_channel,out_channel,stride=1):
    return nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=stride,bias=False)
def con3x3(in_channel,out_channel,stride=1,groups=1,dilation=1):
    """
    padding to ensure the size remains the same
    """
    return nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=stride,padding=dilation,groups=groups,bias=False,dilation=dilation)

class BasicBlock(nn.Module):
    """
    两层连接，论文里对应34层以下的网络
    """
    expansion = 1
    def __init__(self,in_channel,out_channel,stride=1,downsample=None,groups=1,
                 base_width=64,dilation=1,norm_layer=None):
        super(BasicBlock,self).__init__()
        self.conv1 = con3x3(in_channel,out_channel,stride)
        self.bn1 = norm_layer(out_channel)       # 无bn
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = con3x3(out_channel,out_channel)
        self.bn2 = norm_layer(out_channel)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self,x):
        identity = x             # input as the identity
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:       #若downsample为True（默认为False），则对输入执行下采样
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return  out


class Bottleneck(nn.Module):
    expansion = 4
    """
        对特征图的通道先压缩再放大
    """
    def __init__(self,in_channel,out_channel,stride=1,downsample=None,groups=1,
                 base_width=64,dilation=1,norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d                                         # 若未指定，则进行初始化
            
        width = int(out_channel*(base_width/64.))*groups
        
        self.conv1 = conv1x1(in_channel,width)
        self.bn1 = norm_layer(width)                                   #  ?
        self.conv2 = con3x3(width,width,stride,groups,dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width,out_channel * self.expansion)
        self.bn3 = norm_layer(out_channel * self.expansion)         # outchannel is expansion times

        self.relu = nn.ReLU(inplace=True)
        self.dowmsample = downsample
        self.stride = stride

    def forward(self,x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.dowmsample is not None:
            identity = self.dowmsample(x)

        out += identity
        out = self.relu(out)

        return out




class Resnet(nn.Module):
    def __init__(self,block,layers,num_class = 1000,zero_init_residual= False,
                 groups=1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=None):
        super(Resnet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer                         # 访问限制

        self.in_channel = 64                                  # the input of the layer after layer0(conv1)
        self.dilation =1
        if replace_stride_with_dilation is None:
            """
            指明是否用dilaation来代替stride
            """
            replace_stride_with_dilation = [False,False,False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_with = width_per_group

        self.conv1 = nn.Conv2d(3,self.in_channel,kernel_size=7,stride=2,padding=3,bias=False)  # out_size = (x+1)/2
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=)                          # downsample


    def _make_layers(self,block,out,blocks,stride=1,dilate =False):
        """

        :param block:  input one of the blocks(basic or boottle)
        :param out:
        :param blocks:
        :param stride:
        :param dilate:
        :return:
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.in_channel != out * block.expansion:                # neeed downsample
            downsample = nn.Sequential(
                conv1x1(self.in_channel,out*block.expansion,stride),
                norm_layer(out*block.expansion)
            )
        layers = []
        layers.append(block(self.in_channel))



