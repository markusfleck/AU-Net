#!/usr/bin/env python
# coding: utf-8

import torch



class Down_Conv(torch.nn.Module):
            def __init__(self, in_dims, allow_crop = True, dropout = 0.0, batchnorm = True):
            
                assert in_dims[1] == in_dims[2]  # for now, only allow quadratic input images
                assert (in_dims[1] % 4, in_dims[2] % 4) == (0, 0)
                super(Down_Conv, self).__init__()
                
                self.in_channels, self.in_img_dim1, self.in_img_dim2 = in_dims
                self.allow_crop = allow_crop
                self.dropout = dropout
                self.batchnorm = batchnorm
                
                self.crop = ( (self.in_img_dim1 // 4) % 2 == 1)
                
                self.batchnorm1 = torch.nn.BatchNorm2d(self.in_channels)
                self.conv1 = torch.nn.Conv2d(self.in_channels, self.in_channels * 2, kernel_size = 3, padding = 1)
                self.batchnorm2 = torch.nn.BatchNorm2d(self.in_channels * 2)
                self.conv2 = torch.nn.Conv2d(self.in_channels * 2, self.in_channels  * 2, kernel_size = 3, padding = 1)
                self.maxpool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
                
                self.conv_out_dims = (self.in_channels * 2, self.in_img_dim1, self.in_img_dim2)
                
                self.out_dims = (self.in_channels * 2, self.in_img_dim1 // 2, self.in_img_dim2 // 2)
                if (self.crop and self.allow_crop):
                    self.out_dims = (self.out_dims[0], self.out_dims[1] - 2, self.out_dims[2] - 2)
                    
                self.activation = torch.nn.ReLU()
                self.dropout_layer = torch.nn.Dropout(self.dropout)
            
            def forward(self, x):
                if self.batchnorm:
                    x = self.batchnorm1(x)
                x = self.activation(self.conv1(x))
                if self.dropout > 0.0:
                    x = self.dropout_layer(x)
                if self.batchnorm:
                    x = self.batchnorm2(x)
                c = self.activation(self.conv2(x))
                x = self.maxpool(c)
                
                if(self.crop and self.allow_crop):
                    return c, x[:, :, 1:-1, 1:-1 ]
                return c, x


            
class Up_Conv(torch.nn.Module):
            def __init__(self, in_dims, pad = False, dropout = 0.0, batchnorm = True):
                
                super(Up_Conv, self).__init__()
                self.in_channels, self.in_img_dim1, self.in_img_dim2 = in_dims
                self.pad = pad
                self.dropout = dropout
                self.batchnorm = batchnorm
                
                self.up_conv = torch.nn.ConvTranspose2d(self.in_channels, self.in_channels // 2, kernel_size = 2, stride = 2, padding = 0)
                self.batchnorm1 = torch.nn.BatchNorm2d(self.in_channels)
                self.conv1 = torch.nn.Conv2d(self.in_channels, self.in_channels // 2, kernel_size = 3, padding = 1)
                self.batchnorm2 = torch.nn.BatchNorm2d(self.in_channels // 2)
                self.conv2 = torch.nn.Conv2d(self.in_channels // 2, self.in_channels  // 2, kernel_size = 3, padding = 1)
                
                self.out_dims = (self.in_channels // 2, self.in_img_dim1 * 2, self.in_img_dim2 * 2)
                if (self.pad):
                    self.out_dims = (self.out_dims[0], self.out_dims[1] + 4, self.out_dims[2] + 4)
                    
                self.activation = torch.nn.ReLU()
                self.dropout_layer = torch.nn.Dropout(self.dropout)
                    
            def forward(self, concat_layer, x):
                    if self.pad:
                        x = torch.nn.ZeroPad2d(1)(x)
                    x = self.up_conv(x)
                    x = torch.cat([concat_layer, x], axis=1)
                    if self.batchnorm:
                        x = self.batchnorm1(x)
                    x = self.activation(self.conv1(x))
                    if self.dropout > 0.0:
                        x = self.dropout_layer(x)
                    if self.batchnorm:
                        x = self.batchnorm2(x)
                    x = self.activation(self.conv2(x))
                    
                    return x
            
                

class AUnet(torch.nn.Module):
    def __init__(self, depth = 4, input_dims = (3, 256, 256), top_channels = 64, out_channels = 1, dropout = 0.0, batchnorm = True):
        
        assert input_dims[1] == input_dims[2] # for now, only allow quadratic input images
        assert top_channels % 2 == 0
        assert depth > 0
        
        super(AUnet, self).__init__()
        self.depth = depth
        self.top_channels = top_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.batchnorm = batchnorm
        
        
        self.input_adapter_conv = torch.nn.Conv2d(input_dims[0], top_channels // 2, kernel_size = 1)
        
        
        self.down_convs = []
        in_dims = (top_channels // 2, input_dims[1], input_dims[2]) 
        for i in range(self.depth - 1):
            self.down_convs.append(Down_Conv(in_dims, dropout = dropout))
            in_dims = self.down_convs[-1].out_dims
        self.down_convs.append(Down_Conv(in_dims, allow_crop = False, dropout = dropout))
        in_dims = self.down_convs[-1].out_dims
        
        self.batchnorm1 = torch.nn.BatchNorm2d(in_dims[0])
        self.connector_conv1 = torch.nn.Conv2d(in_dims[0], in_dims[0] * 2, kernel_size = 3, padding = 1)
        in_dims = (in_dims[0] * 2, in_dims[1], in_dims[2])
        self.batchnorm2 = torch.nn.BatchNorm2d(in_dims[0])
        self.connector_conv2 = torch.nn.Conv2d(in_dims[0], in_dims[0], kernel_size = 3, padding = 1)
        
        
        self.up_convs = []
        for i in range(self.depth):
            assert (self.down_convs[-i-1].conv_out_dims[1] == in_dims[1] * 2) or (self.down_convs[-i-1].conv_out_dims[1] == in_dims[1] * 2 + 4)
            pad = self.down_convs[-i-1].conv_out_dims[1] == in_dims[1] * 2 + 4
            self.up_convs.append(Up_Conv(in_dims, pad, dropout = dropout))
            in_dims = self.up_convs[-1].out_dims
        
        self.head_conv = torch.nn.Conv2d(in_dims[0], out_channels, kernel_size = 1, padding = 0)
        
        self.activation = torch.nn.ReLU()
        self.dropout_layer = torch.nn.Dropout(self.dropout)
        
            
    def forward(self, x):
        
        x1 = self.input_adapter_conv(x)
        
        
        concats = []
        for i in range(self.depth):
            x0, x1 = self.down_convs[i](x1)
            concats.append(x0)
        
        if self.batchnorm:
            x1 = self.batchnorm1(x1)
        x = self.activation(self.connector_conv1(x1))
        if self.dropout > 0.0:
            x = self.dropout_layer(x)
        if self.batchnorm:
            x = self.batchnorm2(x)
        x = self.activation(self.connector_conv2(x))
        
        
        for i in range(self.depth):
            x = self.up_convs[i](concats[-i-1], x)
            
        
        x = self.head_conv(x)
        
        
        return x



img = torch.rand((17, 3, 300, 300))

for depth in range(4,5): 
    aunet = AUnet(depth = depth, input_dims = (3, 300, 300), top_channels = 64, out_channels = 3, dropout = 0.1, batchnorm = True)
    out = aunet(img)
    print('out.shape', out.shape)






