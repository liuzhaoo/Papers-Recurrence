import torch.nn as nn
import torch

# __all__ = ['resnet34', 'resnet50', 'resnet101']


# 首先定义bottleneck块的1x1 和3x3 卷积

def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)


def con3x3(in_channel, out_channel, stride=1, groups=1, dilation=1):
    """
    padding to ensure the size remains the same
    """
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    """
    两层连接，论文里对应34层以下的网络
    """
    expansion = 1

    def __init__(self, inchannel, outchannel, stride=1, downsample=None, groups=1,
                  dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # if groups != 1 or base_width != 16:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = con3x3(inchannel, outchannel, stride)
        self.bn1 = norm_layer(outchannel)  # 无bn
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = con3x3(outchannel, outchannel)
        self.bn2 = norm_layer(outchannel)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x  # input as the identity

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        a = out.size()
        b_= identity.size


        if self.downsample is not None:  # 若downsample为True（默认为False），则对输入执行下采样
            identity = self.downsample(x)
        b = identity.size()
        out += identity
        out = self.relu(out)
        c = out.size()
        return out





class Resnet(nn.Module):
    def __init__(self, layers, num_class=10, zero_init_residual=False,
                 groups=1, replace_stride_with_dilation=None, norm_layer=None):
        super(Resnet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer  # 访问限制

        self.in_channel = 16  # the input of the layer after layer0(conv1)16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            """
            指明是否用dilaation来代替stride
            """
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        # self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=3, stride=1, padding=1, bias=False)  # out_size = (x+1)/2
        self.bn1 = norm_layer(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)                          # downsample

        self.layer1 = self._make_layers(16, layers[0])  # layers is a list ,including the numbers of blocks
        self.layer2 = self._make_layers(32, layers[1], stride=2,
                                        dilate=replace_stride_with_dilation[
                                            0])  # since the layer2, apply downsample in the first block

        self.layer3 = self._make_layers(64, layers[2], stride=2,
                                        dilate=replace_stride_with_dilation[1])

        # self.layer4 = self._make_layers(block,512, layers[3], stride=2,
        #                                 dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * BasicBlock.expansion, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        m = out.size()
        out = self.layer1(out)
        s1 = out.size()
        out = self.layer2(out)
        s2 = out.size()
        out = self.layer3(out)
        # out = self.layer4(out)
        s3 =out.size()
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    def _make_layers(self, out, blocks, stride=1, dilate=False):
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

        if stride != 1 or self.in_channel != out * BasicBlock.expansion:  # neeed downsample
            downsample = nn.Sequential(
                conv1x1(self.in_channel, out * BasicBlock.expansion, stride),
                norm_layer(out * BasicBlock.expansion),
            )
        layers = []
        layers.append(BasicBlock(self.in_channel, out, stride, downsample, self.groups,
                            previous_dilation, norm_layer))

        self.in_channel = out * BasicBlock.expansion  # update the input channel for next layer
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channel, out, groups=self.groups,
                                 dilation=previous_dilation, norm_layer=norm_layer))
        # 记录问题： 这里传入的参数中，stride应为默认的1，才会在每个layer的第二个block以后不进行通道减半
        return nn.Sequential(*layers)


def _resnet(layers, **kwargs):
    model = Resnet(layers, **kwargs)

    return model


def resnet56(**kwargs):
    return _resnet([12, 12, 12], **kwargs)
