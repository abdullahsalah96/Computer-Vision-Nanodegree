## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        self.pool = nn.MaxPool2d(2,2)
        
        #input image dims is 224x224
        self.conv1 = nn.Conv2d(1, 32, 5)            #after 1st conv+pooling image dims = ( (224-4)/2 , (224-4)/2 ) = (110,110)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv1_drop = nn.Dropout(0.6)         
        
        self.conv2 = nn.Conv2d(32, 64, 5)           #after 2nd conv+pooling image dims = ( (110-4)/2 , (110-4)/2 ) = (53,53)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv2_drop = nn.Dropout(0.5)
        
        self.conv3 = nn.Conv2d(64, 128, 5)          #after 3rd conv+pooling image dims = ( (53-4)/2 , (53-4)/2 ) = (24,24)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv3_drop = nn.Dropout(0.4)

        self.conv4 = nn.Conv2d(128, 256, 5)         #after 4th conv+pooling image dims = ( (24-4)/2 , (24-4)/2 ) = (10,10)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.conv4_drop = nn.Dropout(0.3)

        self.fc1 = nn.Linear(256*10*10, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc1_drop = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(512,256)               
        self.fc2_bn = nn.BatchNorm1d(256)
        self.fc2_drop = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(256,136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv2_bn(x)
        x = self.conv2_drop(x)
        
        x = self.pool(F.relu(self.conv3(x)))
        x = self.conv3_bn(x)
        x = self.conv3_drop(x)
        
        x = self.pool(F.relu(self.conv4(x)))
        x = self.conv4_bn(x)
        x = self.conv4_drop(x)
        
        x = x.view(x.size(0), -1) #Flatten

        x = F.relu(self.fc1(x))
        x = self.fc1_bn(x)
        x = self.fc1_drop(x)
        
        x = F.relu(self.fc2(x))
        x = self.fc2_bn(x)
        x = self.fc2_drop(x)
        
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
