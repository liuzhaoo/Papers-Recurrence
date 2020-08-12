import math
import torch.nn as nn
import torch

__all__ = ['resnet34','resnet50','resnet101']


# 首先定义bottleneck块的1x1 和3x3 卷积

def conv1x1(in_place,out_place,stride=1):
    return nn.Conv2d(in_place,out_place,kernel_size=1,stride=stride,bias=False)
def con3x3(in_place,out_place,stride=1,groups=1,dilation=1):
    """
    padding to ensure the size remains the same
    """
    return nn.Conv2d(in_place,out_place,kernel_size=3,stride=stride,padding=dilation,groups=groups,bias=False,dilation=dilation)

class BasicBlock(nn.Module):
    """
    两层连接，论文里对应34层以下的网络
    """
    expansion = 1
    def __init__(self,inplace,outplace,stride=1,downsample=None,groups=1,
                 base_width=64,dilation=1,norm_layer=None):
        super(BasicBlock,self).__init__()
        self.conv1 = con3x3(inplace,outplace,stride)
        self.bn1 = norm_layer(outplace)       # 无bn
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = con3x3(outplace,outplace)
        self.bn2 = norm_layer(outplace)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self,x):
        identity = x             # input as the identity
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return  out