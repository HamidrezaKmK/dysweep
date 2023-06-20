import torch
from torch import nn
import typing as th

class CustomConvNet(nn.Module):
    def __init__(self, num_classes, in_channels, in_height, in_width, 
                 fully_connected_layer_sizes: th.Optional[th.List[int]] = None):
        super(CustomConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # calculate the size of the feature map after two rounds of convolution and pooling
        self.feature_map_height = int(in_height/2/2)
        self.feature_map_width = int(in_width/2/2)
        
        if fully_connected_layer_sizes is None:
            fully_connected_layer_sizes = [1024, num_classes]
        else:
            fully_connected_layer_sizes += [num_classes]
            
        cur_sz = 64*self.feature_map_height*self.feature_map_width
        for i, layer_size in enumerate(fully_connected_layer_sizes):
            self.register_module(f'fc{i}', nn.Linear(cur_sz, layer_size))
            cur_sz = layer_size
            if i < len(fully_connected_layer_sizes) - 1:
                self.register_module(f'relu{i + 2}', nn.ReLU())
        self.layer_sizes = fully_connected_layer_sizes
                
       
    def forward(self, x):
        # output the logits
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        for i, layer_size in enumerate(self.layer_sizes):
            x = getattr(self, f'fc{i}')(x)
            if i < len(self.layer_sizes) - 1:
                x = getattr(self, f'relu{i + 2}')(x)
        return x
