import cv2
import numpy as np
import torch
import torch.nn as nn
import random
import glob
from PIL import Image
from torch.utils.data import Dataset

# from tensorflow import keras

class BatchLoader(Dataset):
    def __init__(self, datasetRoot, imgHeight=32, imgWidth=32, shuffle=False):
        super(BatchLoader, self).__init__()

        self.path = datasetRoot #dataset root path
        self.imgHeight = imgHeight #img crop height
        self.imgWidth = imgWidth #img width height
        self.imgNames = []

        classList = glob.glob(self.path + "/*")
        # print(classList)
        for img_path in classList: #for each class
            self.imgNames.append(img_path)

        self.count = len(self.imgNames)
        self.perm = list(range(self.count))

        if shuffle == True:
            random.shuffle(self.perm)

        self.itercount = 0

    def __len__(self):
        return self.count

    def __getitem__(self, idx):

        imName = self.imgNames[self.perm[idx]]
        # imPath = '/' + imName.split("/")[-2] + '/' + imName.split("/")[-1]
        im = self.loadImage(imName)
#         plt.imshow(im.transpose(1,2,0))
        return im, imName


    def loadImage(self, imName):
        im = Image.open(imName)
        im = np.asarray(im)

        # rows, cols = im.shape[0], im.shape[1]
        im = cv2.resize(im, (self.imgHeight, self.imgWidth), interpolation=cv2.INTER_LINEAR)

        if len(im.shape) == 2:
            # print('Warning: load a gray image')
            im = im[:, :, np.newaxis]
            im = np.concatenate([im, im, im], axis=2)
        elif len(im.shape) > 2 and im.shape[2] == 4:
            # print("Warning: load a RGBA image")
            im = im[:,:,:3]

        im = im.astype(np.uint8) #/ 255.0
        
#         print(type(im))
        im = im.transpose([2, 0, 1] )
#         plt.imshow(im)
        return im

# class createAugment(keras.utils.Sequence):
#     'Generates data for Keras'
#     def __init__(self, X, y, batch_size=6, imgHeight=32, imgWidth=32, n_channels=3, shuffle=True, thickness=3):
#         'Initialization'
#         self.batch_size = batch_size 
#         self.X = X 
#         self.y = y
#         self.thickness = thickness
#         self.imgHeight = imgHeight
#         self.imgWidth = imgWidth
#         self.n_channels = n_channels
#         self.shuffle = shuffle
#         self.indexes = np.arange(len(self.X))
#         if self.shuffle:
#             np.random.shuffle(self.indexes)

#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return int(np.floor(len(self.X) / self.batch_size))

#     def __getitem__(self, index):
#         'Generate one batch of data'
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

#         # Generate data
#         return self.__data_generation(indexes)

#     def __data_generation(self, idxs):
#         # X_batch is a matrix of masked images used as input
#         X_batch = np.empty((self.batch_size, self.n_channels, self.imgHeight, self.imgWidth))
# #         X_batch = np.empty((self.batch_size, self.imgHeight, self.imgWidth, self.n_channels))# Masked image
#         # y_batch is a matrix of original images used for computing error from reconstructed image
#         y_batch = np.empty((self.batch_size, self.n_channels, self.imgHeight, self.imgWidth)) # Original image
# #         y_batch = np.empty((self.batch_size, self.imgHeight, self.imgWidth, self.n_channels))
#         mask_batch = np.empty((self.batch_size, self.n_channels, self.imgHeight, self.imgWidth))
#         ## Iterate through random indexes
#         for i, idx in enumerate(idxs):
#             image_copy = self.X[idx].copy()

#             ## Get mask associated to that image
#             masked_image, mask = self.__createMask(image_copy)

#             X_batch[i,] = masked_image/255
#             y_batch[i] = self.y[idx]/255
#             mask_batch[i] = mask / 255
            
# #             print("masked_image shape: ", masked_image.shape)
# #             print("mask\ shape: ", mask.shape)
# #             print("orig shape: ", self.y[idx].shape)

#         return X_batch, y_batch, mask_batch

#     def __createMask(self, img):
#         ## Prepare masking matrix
#         mask = np.full((self.imgHeight,self.imgWidth,3), 255, np.uint8)
#         for _ in range(np.random.randint(1, 10)):
#             # Get random x locations to start line
#               x1, x2 = np.random.randint(1, self.imgHeight), np.random.randint(1, 256)
#               # Get random y locations to start line
#               y1, y2 = np.random.randint(1, self.imgWidth), np.random.randint(1, 256)
#               # Get random thickness of the line drawn
#               thickness = np.random.randint(1, self.thickness)
#               # Draw black line on the white mask
#               cv2.line(mask,(x1,y1),(x2,y2),(0,0,0),thickness)

#         # Perforn bitwise and operation to mak the image
#         masked_image = cv2.bitwise_and(img.transpose(1,2,0), mask)
#         masked_image = masked_image.transpose(2,0,1)
#         mask = mask.transpose(2,0,1)

#         return masked_image , mask

