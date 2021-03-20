import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class ModelBuilder():
    def build_net(self, num_input=4, num_classes=5, num_branches=4, padding_list=[0,4,8,12], dilation_list=[2,6,10,14], input_shape=[4,160, 192, 128]):
        # parameters in the architecture
        channels= [num_input-1, 32, 64, 128, 256, 128, 64, 32, num_classes]
        kernel_size = 3
        network = relax_focus(channels, kernel_size, padding_list, dilation_list, num_branches)


        return network


class relax_focus(nn.Module):
    def __init__(self, channels, kernel_size,padding_list, dilation_list, num_branches):
        super(relax_focus, self).__init__()
        self.channels = channels
        self.padding_list = padding_list
        self.dilation_list = dilation_list
        self.num_branches = num_branches
        self.block1 = Conv3d_Block(self.channels[0], self.channels[1],self.channels[1])
        self.auto1 = Autofocus_single(self.channels[1],self.channels[1],self.channels[1],self.padding_list, self.dilation_list, self.num_branches)
        #
        self.block_y1 = Conv3d_Block(self.channels[0]*2, self.channels[1],self.channels[1])
        self.auto_y1 = Autofocus_single(self.channels[1],self.channels[1],self.channels[1],self.padding_list, self.dilation_list, self.num_branches)

        self.block2 = ResBlock(int(self.channels[1]), self.channels[2], self.channels[2])
        self.auto2 = Autofocus_single(self.channels[2],self.channels[2],self.channels[2],self.padding_list, self.dilation_list, self.num_branches)

        self.block_y2 = ResBlock(int(self.channels[1]), self.channels[2], self.channels[2])
        self.auto_y2 = Autofocus_single(self.channels[2],self.channels[2],self.channels[2],self.padding_list, self.dilation_list, self.num_branches)

        self.block3 = ResBlock(int(self.channels[2]), self.channels[3], self.channels[3])
        self.auto3 = Autofocus_single(self.channels[3],self.channels[3],self.channels[3],self.padding_list, self.dilation_list, self.num_branches)

        self.block_y3 = ResBlock(int(self.channels[2]), self.channels[3], self.channels[3])
        self.auto_y3 = Autofocus_single(self.channels[3],self.channels[3],self.channels[3],self.padding_list, self.dilation_list, self.num_branches)

        self.block4 = ResBlock(int(self.channels[3]), self.channels[3], self.channels[3])
        self.block4_1 = Conv3d(self.channels[5],self.channels[6])
        self.block4_2 = Conv3d(self.channels[6],self.channels[7])
        self.block5_1 = Conv3d(self.channels[6],self.channels[7])
        self.block5 = ResBlock(int(self.channels[4]), self.channels[5],self.channels[6])
        self.auto5 = Autofocus_single(self.channels[6],self.channels[6],self.channels[6],self.padding_list, self.dilation_list, self.num_branches)

        self.block6 = ResBlock(int(self.channels[5]), self.channels[6],self.channels[7])
        self.auto6 = Autofocus_single(self.channels[7],self.channels[7],self.channels[7],self.padding_list, self.dilation_list, self.num_branches)

        self.block7 = ResBlock(int(self.channels[6]), self.channels[7],self.channels[7])
        self.auto7 = Autofocus_single(self.channels[7],self.channels[7],self.channels[7],self.padding_list, self.dilation_list, self.num_branches)

        self.pool = nn.MaxPool3d(kernel_size=2,return_indices = True)
        self.unpool=nn.MaxUnpool3d(kernel_size=2)

        self.block8 =Conv3d_Block(self.channels[7]*4, self.channels[7]*2,self.channels[7])
        self.fc =  nn.Conv3d(self.channels[7], self.channels[8], kernel_size=1)
        self.atten = nn.Softmax(dim = 1)
        self.dropout = nn.Dropout3d(p=0.5)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, y):

        y = torch.cat((x1, y),1)

        x1 = self.block1(x1)
        x1 = self.auto1(x1)
        x2,x2_indices = self.pool(x1)

        x2 = self.block2(x2)
        x2 = self.auto2(x2)
        x3,x3_indices = self.pool(x2)

        y = self.block_y1(y)
        y = self.auto_y1(y)
        y, _ = self.pool(y)

        y = self.block_y2(y)
        y = self.auto_y2(y)
        y, _ = self.pool(y)

        y = self.block_y3(y)
        y = self.auto_y3(y)
        y, _ = self.pool(y)

        x3 = x3 *self.atten(y)

        x3 = self.block3(x3)
        x3 = self.auto3(x3)
        x4, x4_indices = self.pool(x3)

        x4 = self.block4(x4)
        x4 = self.dropout(x4)
        x4=self.unpool(x4,x4_indices)



        y4 =self.block4_1(x4)
        y4 = self.unpool(y4,x3_indices)
        y4 = self.block4_2(y4)
        y4 = self.unpool(y4,x2_indices)


        x4=self.block5(torch.cat((x4,x3),1))
        x4= self.auto5(x4)


        x4=self.unpool(x4,x3_indices)

        y5 = self.block5_1(x4)
        y5 = self.unpool(y5,x2_indices)


        x4=self.block6(torch.cat((x4,x2),1))
        x4 = self.auto6(x4)

        x4=self.unpool(x4,x2_indices)



        x5= self.block7(torch.cat((x5,x1),1))
        x5 = self.auto7(x5)
        x5 = self.block8(torch.cat((x5,y6,y5,y4),dim = 1))
        x5 = self.fc(x5)
        return x4


class Conv3d_Block(nn.Module):
    def __init__(self, inplanes, outplanes1, outplanes2, kernel = 3, padding = True, d_rate=1):
        super(Conv3d_Block, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, outplanes1, kernel_size=kernel, dilation = d_rate)
        self.bn1 = nn.BatchNorm3d(outplanes1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(outplanes1,outplanes2, kernel_size=kernel, dilation = d_rate)
        self.bn2 = nn.BatchNorm3d(outplanes2)
        self.padding = padding
        self.padding_f = nn.ReplicationPad3d(1)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        if self.padding:
            x = self.padding_f(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.padding:
            x = self.padding_f(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, inplanes, outplanes1, outplanes2, kernel=3, num_groups=4, downsample=None, padding = True,d_rate = 1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, outplanes1, kernel_size=kernel,dilation = d_rate, padding = np.int(d_rate*(kernel-1)/2), padding_mode= 'replicate')
        self.bn1 = nn.BatchNorm3d(outplanes1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(outplanes1,outplanes2, kernel_size=kernel,dilation = d_rate, padding = np.int(d_rate*(kernel-1)/2), padding_mode= 'replicate')
        self.bn2 = nn.BatchNorm3d(outplanes2)

        if inplanes==outplanes2:
            self.downsample = downsample
        else:
            self.downsample = nn.Sequential(nn.Conv3d(inplanes, outplanes2, kernel_size=1), nn.BatchNorm3d(outplanes2))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        residual = x
        print (x.size())
        x = self.conv1(x)
        print (x.size())
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)
        return x

class Autofocus_single(nn.Module):
    def __init__(self, inplanes1, outplanes1, outplanes2, padding_list, dilation_list, num_branches, kernel=3):
        super(Autofocus_single, self).__init__()
        self.padding_list = padding_list
        self.dilation_list = dilation_list
        self.num_branches = num_branches
        self.conv1 =nn.Conv3d(inplanes1, outplanes1, kernel_size=kernel,padding=self.padding_list[0], dilation=2)
        self.bn1 = nn.BatchNorm3d(outplanes1)
        self.padding = nn.ReplicationPad3d(1)
        self.padding2 = nn.ReplicationPad3d(2)

        self.bn_list2 = nn.ModuleList()
        for i in range(len(self.padding_list)):
            self.bn_list2.append(nn.BatchNorm3d(outplanes2))

        self.conv2 =nn.Conv3d(outplanes1, outplanes2, kernel_size=kernel, padding=self.padding_list[0],dilation=self.dilation_list[0])
        self.convatt1 = nn.Conv3d(outplanes1, int(outplanes1/2), kernel_size=kernel)
        self.convatt2 = nn.Conv3d(int(outplanes1/2), self.num_branches, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        if inplanes1==outplanes2:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv3d(inplanes1, outplanes2, kernel_size=1), nn.BatchNorm3d(outplanes2))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x
        x = self.conv1(self.padding2(x))
        x = self.bn1(x)
        x = self.relu(x)


        # compute attention weights for the second layer
        feature = x.detach()
        att = self.relu(self.convatt1(self.padding(feature)))
        att = self.convatt2(att)
        att = F.softmax(att, dim=1)

        # linear combination of different dilation rates
        x1 = self.conv2(self.padding2(x))
        shape = x1.size()
        x1 = self.bn_list2[0](x1)* att[:,0:1,:,:,:].expand(shape)

        # sharing weights in parallel convolutions
        for i in range(1, self.num_branches):
            x = self.padding2(x)
            x2 = F.conv3d(x, self.conv2.weight, padding=self.padding_list[i], dilation=self.dilation_list[i])
            x2 = self.bn_list2[i](x2)
            x1 += x2* att[:,i:(i+1),:,:,:].expand(shape)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x = x1 + residual
        x = self.relu(x)
        return x
