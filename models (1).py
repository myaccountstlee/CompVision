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
        # Color image has 3 inputs. 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #( W-F)/S + 1 = (224-5)/1 + 1 = 220
        # Output dimensions 32 x 220 x 220
        # After 3 x 3 Maxpool layer this becomes (220/2) = 110.   32 x 110 x 110
        self.conv1 = nn.Conv2d(1, 32, 5)
                    
        self.pool = nn.MaxPool2d(2, 2)
        
        # dropout with p=0.1
        self.conv1_drop = nn.Dropout(p=0.1)
        
        #Another conv layer
        #32 outputs, kernal size 3, stride 1
        # (W-F)/S + 1 = (110-3)/1 + 1 = 108
        # Output dimensions 64 X 108 x 108
        # After 2 x 2 Maxpool layer this becomes 108/2 = 54.  64 x 54 x 54
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        self.conv2_bn = nn.BatchNorm2d(64)  #batch normalization
        
        # dropout with p=0.4
        self.conv2_drop = nn.Dropout(p=0.2)
        
        # 64 outputs, Kernal size 2, stride 1
        # (54 - 3)/1 + 1 = 52
        # Output dimensions 128 x 52 x 52
        # After 2 x 2 Maxpool layer this becomes 52/2 = 26.  128 x 26 x 26
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        # dropout with p=0.3
        self.conv3_drop = nn.Dropout(p=0.26)
                
        # (26 - 3)/1 + 1 = 24
        # Output dimensions 256 x 24 x 24
        # After 2 x 2 Maxpool layer:  256 x 12 x 12
        self.conv4 = nn.Conv2d(128, 256, 3)
        
        self.conv4_drop = nn.Dropout(p=0.26)
        
        # 512 outputs, Kernal size 3, stride 1
        # (12 - 1)/1 + 1 = 12
        # Output dimensions 512 x 12 x 12
        # After 2 x 2 Maxpool layer:  512 x 6 x 6
        self.conv5 = nn.Conv2d(256, 512, 1)
        
        self.conv5_drop = nn.Dropout(p=0.3)
        
        # 1024 outputs, Kernal size 1, stride 1
        # (5 - 1)/1 + 1 = 5
        # Output dimensions 1024 x 5 x 5
        # After 2 x 2 Maxpool layer:  512 x 2 x 2
        #self.conv6 = nn.Conv2d(512, 1024, 1)
        
        self.fc1 = nn.Linear(512*6*6,1024)
        
        self.fc1_bn = nn.BatchNorm1d(1024)  #batch normalization
        
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)
        
        # finally, create 136 output channels (two outputs for each keypoint x,y pair)
        self.fc2 = nn.Linear(1024, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        
        #Dropout layer after convolutional layer 1
        x = self.conv1_drop(x)
            
        x = self.pool(F.relu(self.conv2_bn((self.conv2(x)))))
        
        #Dropout layer after convolutional layer 2
        x = self.conv2_drop(x)
        
        x = self.pool(F.relu(self.conv3(x)))
            
        #Dropout layer after convolutional layer 3
        x = self.conv3_drop(x)
        
        x = self.pool(F.relu(self.conv4(x)))
        
        #Dropout layer after convolutional layer 4
        x = self.conv4_drop(x)
        
        x = self.pool(F.relu(self.conv5(x)))
        
        #Dropout layer after convolutional layer 5
        x = self.conv5_drop(x)
     
        
        #Prep for Linear layer
        x = x.view(x.size(0), -1)
        
        # print('lin_prep: ' + str(x.shape))
        
        #Linear layer
        x = F.relu(self.fc1_bn((self.fc1(x))))
        # print('fc1: ' + str(x.shape))
        x = self.fc1_drop(x)
        # print('fc2: ' + str(x.shape))
        x = self.fc2(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
