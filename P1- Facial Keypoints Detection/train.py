import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Net
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from load_data import FacialKeypointsDataset
from load_data import Rescale, RandomCrop, Normalize, ToTensor
import os
import cv2
import torch.optim as optim

net = Net()
print(net)

data_transform = transforms.Compose([Rescale(250), RandomCrop(224), Normalize(), ToTensor()])
# create the transformed dataset
key_pts_path = '/Users/abdallaelshikh/Documents/Computer Vision Nanodegree/Facial Keypoints/data/train-test-data/training_frames_keypoints.csv'

transformed_dataset = FacialKeypointsDataset(csv_file=key_pts_path,
                                             root_dir=os.path.dirname(os.path.realpath(__file__)) + '/data/train-test-data/training/',
                                             transform=data_transform)

print('Number of images: ', len(transformed_dataset))

# sample = transformed_dataset[0]
# img = sample['image'].numpy()
# img = np.reshape(np.uint8(img*255.0), (224,224,1))
# pts = sample['keypoints'].numpy()*50.0+100
# print(type(pts))
# print(pts[0])
# for pt in pts:
#     cv2.circle(img, (int(pt[0]), int(pt[1])),1,(0,255,0),1)
# cv2.imshow('img', img)
# cv2.waitKey(0)

# load training data in batches
# batch_size = 10
# train_loader = DataLoader(transformed_dataset, 
#                           batch_size=batch_size,
#                           shuffle=True, 
#                           num_workers=4)

# # create the test dataset
# test_dataset = FacialKeypointsDataset(csv_file=key_pts_path,
#                                       root_dir=os.path.dirname(os.path.realpath(__file__)) + '/data/train-test-data/test/',
#                                       transform=data_transform)

# test_loader = DataLoader(test_dataset, 
#                           batch_size=batch_size,
#                           shuffle=True, 
#                           num_workers=4)


# criterion = nn.SmoothL1Loss()
# optimizer = optim.Adam(net.parameters(), lr=0.001)

# def train_net(n_epochs):
#     # prepare the net for training
#     net.train()
#     for epoch in range(n_epochs):  # loop over the dataset multiple times
#         running_loss = 0.0
#         # train on batches of data, assumes you already have train_loader
#         for batch_i, data in enumerate(train_loader):
#             # get the input images and their corresponding labels
#             images = data['image']
#             key_pts = data['keypoints']
#             # flatten pts
#             key_pts = key_pts.view(key_pts.size(0), -1)
#             # convert variables to floats for regression loss
#             key_pts = key_pts.type(torch.FloatTensor)
#             images = images.type(torch.FloatTensor)
#             # forward pass to get outputs
#             output_pts = net(images)
#             # calculate the loss between predicted and target keypoints
#             loss = criterion(output_pts, key_pts)
#             # zero the parameter (weight) gradients
#             optimizer.zero_grad()
#             # backward pass to calculate the weight gradients
#             loss.backward()
#             # update the weights
#             optimizer.step()
#             # print loss statistics
#             running_loss += loss.item()
#             if batch_i % 10 == 9:    # print every 10 batches
#                 print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/10))
#                 running_loss = 0.0

#     print('Finished Training')

# n_epochs = 5
# train_net(n_epochs)

# ## TODO: change the name to something uniqe for each new model
# model_dir = 'saved_models/'
# model_name = 'face_model.pt'

# # after training, save your model parameters in the dir 'saved_models'
# torch.save(net.state_dict(), model_dir+model_name)