import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_sizes, strides, paddings, batch_norm: bool = False):
        super(ConvBlock, self).__init__()
        self.do_batch_norm = batch_norm
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_sizes, strides, paddings)
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.do_batch_norm:
            x = self.bn(x)
        x = self.relu(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        #y_sorted,indices = t.sort(y)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def SimAM(X):
	# X: input feature [N, C, H, W]
	# lambda: coefficient λ in Eqn (5)
	n = X.shape[2] * X.shape[3] - 1
	# square of (t - u)
	d = (X - X.mean(dim=[2,3]).view(X.size(0),X.size(1),1,1)).pow(2)
	# d.sum() / n is channel variance
	v = d.sum(dim=[2,3]) / n 
	v = v.view(v.size(0),v.size(1),1,1)
	# E_inv groups all importance of X
	#E_inv = d / (4 * (v + lambda)) + 0.5
	E_inv = d / (4 * (v + 0.001)) + 0.5  #set λ=0.0001
	# return attended features
	return X * F.sigmoid(E_inv)	

class HAMVisContexNN(nn.Module):
    def __init__(self, img_channel, num_class, map_to_seq_hidden=64, rnn_hidden=256):
        super(HAMVisContexNN, self).__init__()
        # CNN block
        self.b0_c1 = nn.Conv2d(1, 32, 3, 1,1)
        self.b0_b1 = nn.BatchNorm2d(32)
        self.b0_se = SELayer(32)  #SE
        self.b0_p = nn.MaxPool2d(kernel_size=3, stride=2)
        #Block1
        self.b1_c1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.b1_b1 = nn.BatchNorm2d(32)
        self.b1_c2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.b1_b2 = nn.BatchNorm2d(32)

        self.b1_se = SELayer(32)  #SE
        self.b1_p = nn.MaxPool2d(kernel_size=2, stride=2)
        #Block2
        self.b2_c1 = nn.Conv2d(32, 64, 3, 1, 1)
        self.b2_b1 = nn.BatchNorm2d(64)
        self.b2_c2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.b2_b2 = nn.BatchNorm2d(64)

        self.b2_up1c = nn.Conv2d(32, 64, 1, 1)
        self.b2_up1b = nn.BatchNorm2d(64)

        self.b2_se = SELayer(64)  #SE
        self.b2_p = nn.MaxPool2d(kernel_size=2, stride=2)
        #Block3
        self.b3_c1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.b3_b1 = nn.BatchNorm2d(128)
        self.b3_c2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.b3_b2 = nn.BatchNorm2d(128)
        self.b3_c3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.b3_b3 = nn.BatchNorm2d(128)
        self.b3_c4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.b3_b4 = nn.BatchNorm2d(128)

        self.b3_up1c = nn.Conv2d(64, 128, 1, 1)
        self.b3_up1b = nn.BatchNorm2d(128)

        self.b3_se = SELayer(128)  #SE
        self.b3_p  = nn.MaxPool2d(kernel_size=(2, 1))
        #Block4
        self.b4_c1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.b4_b1 = nn.BatchNorm2d(256)
        self.b4_c2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.b4_b2 = nn.BatchNorm2d(256)
        self.b4_c3 = nn.Conv2d(256, 512, (3,1), 1, 1)
        self.b4_b3 = nn.BatchNorm2d(512)
        self.b4_c4 = nn.Conv2d(512, 512, (2,1), 1, 1)
        self.b4_b4 = nn.BatchNorm2d(512)
        self.b4_se = SELayer(512)  #SE

        self.b4_up1c = nn.Conv2d(128, 256, 1, 1)
        self.b4_up1b = nn.BatchNorm2d(256)

        self.b4_up2c = nn.Conv2d(256, 512, (2,1), 1, (1,2))
        self.b4_up2b = nn.BatchNorm2d(512)

        # map CNN to sequence
        self.map2seq = nn.Linear(512, map_to_seq_hidden)        
	# RNN
        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)
        # fully connected
        self.dense = nn.Linear(1024, num_class)

    def forward(self, x):  

        # CNN block
        out = self.b0_c1(x)
        out = self.b0_b1(out)
        out = nn.ReLU(inplace=True)(out)
        out = SimAM(out)  
        out = self.b0_se(out)  
        out = self.b0_p(out)
        #Block1
        identi1 = out

        out = self.b1_c1(out)
        out = self.b1_b1(out)
        out = nn.ReLU(inplace=True)(out)
        out = SimAM(out)   
        out = self.b1_c2(out)
        out = self.b1_b2(out)

        out = nn.ReLU(inplace=True)(identi1 + out)
        out = SimAM(out)   

        out = self.b1_se(out)  
        out = self.b1_p(out)
        #Block2
        identi2 = self.b2_up1c(out)
        identi2 = self.b2_up1b(identi2)

        out = self.b2_c1(out)
        out = self.b2_b1(out)
        out = nn.ReLU(inplace=True)(out)
        out = SimAM(out)   
        out = self.b2_c2(out)
        out = self.b2_b2(out)

        out = nn.ReLU(inplace=True)(identi2 + out)
        out = SimAM(out)   
        out = self.b2_se(out)  
        out = self.b2_p(out)
        #Block3
        identi31 = self.b3_up1c(out)
        identi31 = self.b3_up1b(identi31)

        out = self.b3_c1(out)
        out = self.b3_b1(out)
        out = nn.ReLU(inplace=True)(out)
        out = SimAM(out)   
        out = self.b3_c2(out)
        out = self.b3_b2(out)

        out = nn.ReLU(inplace=True)(identi31 + out)
        out = SimAM(out)   

        identi32 = out

        out = self.b3_c3(out)
        out = self.b3_b3(out)
        out = nn.ReLU(inplace=True)(out)
        out = SimAM(out)   
        out = self.b3_c4(out)
        out = self.b3_b4(out)

        out = nn.ReLU(inplace=True)(identi32 + out)
        out = SimAM(out)   
        out = self.b3_se(out)  
        out = self.b3_p(out)
        #Block4
        identi41 = self.b4_up1c(out)
        identi41 = self.b4_up1b(identi41)

        out = self.b4_c1(out)
        out = self.b4_b1(out)
        out = nn.ReLU(inplace=True)(out)
        out = SimAM(out)   
        out = self.b4_c2(out)
        out = self.b4_b2(out)

        out = nn.ReLU(inplace=True)(identi41 + out)
        out = SimAM(out)   


        identi42 = self.b4_up2c(out)
        identi42 = self.b4_up2b(identi42)

        out = self.b4_c3(out)
        out = self.b4_b3(out)
        out = nn.ReLU(inplace=True)(out)
        out = SimAM(out)   
        out = self.b4_c4(out)
        out = self.b4_b4(out)



        out = nn.ReLU(inplace=True)(identi42 + out)
        out = SimAM(out)   
        out = self.b4_se(out)  


        # reformat array
        batch, channel, height, width = out.size()
        out = nn.functional.adaptive_avg_pool2d(out,(1,width))
        out = out.view(batch, channel, width)
        out = out.permute(2, 0, 1)
        vis_f = out
        out = self.map2seq(out)
        out, _ = self.rnn1(out)
        out, _ = self.rnn2(out)

        out = torch.cat((out,vis_f),2)
        out = self.dense(out)
        return out

class WIDNN(nn.Module):
    def __init__(self, img_channel, num_class, map_to_seq_hidden=64, rnn_hidden=256):
        super(WIDNN, self).__init__()
        # CNN block
        self.cnn = nn.Sequential(
            ConvBlock(img_channel, 16, 3, 1, 1, batch_norm=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ConvBlock(16, 32, 3, 1, 1,batch_norm=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(32, 64, 3, 1, 1,batch_norm=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 64, 3, 1, 1,batch_norm=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            ConvBlock(64, 64, (3,1), 1, 1, batch_norm=True),
            ConvBlock(64, 128, (2,1), 1, 1, batch_norm=True),
        )
        # map CNN to sequence
        self.map2seq = nn.Linear(128, map_to_seq_hidden)        
	# RNN
        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        # fully connected
        self.dense = nn.Linear(2*rnn_hidden, num_class)

    def forward(self, x):
        # CNN block
        x = self.cnn(x)

        # reformat array
        batch, channel, height, width = x.size()
        x = nn.functional.adaptive_avg_pool2d(x,(1,width))
        x = x.view(batch, channel, width)
        x = x.permute(2, 0, 1)
        x = self.map2seq(x)
        x, _ = self.rnn1(x)
        x = x.permute(1,0,2)
        _,_,feature_num = x.size()
        x = nn.functional.adaptive_avg_pool2d(x,(1,feature_num))
        x = x.view(batch,feature_num)
        x = self.dense(x)
        return x
