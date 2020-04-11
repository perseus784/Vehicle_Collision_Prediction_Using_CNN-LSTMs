import cv2
import os
import json
import random
from config import *
import numpy as np
import h5py

train_folder = os.path.join('datasets','train_set')
test_folder = os.path.join('datasets','test_set')

class utils:

    def __init__(self):
        self.train_files = os.listdir(train_folder)
        self.test_files = os.listdir(test_folder)
            
    def get_sequence(self):
        return 0

    def batch_dispatch(self,batch_size):
        start_index = 0
        end_index = batch_size
        train_data = os.listdir(train_folder)

        while flag:
            image_seqs, labels, flag = self.get_sequence()
            #print('printing here',image_seqs)
            #image_seqs = image_seqs.reshape((batch_size,time,height,width,color_channels))
            print('printing here', image_seqs.shape)
            #labels = np.eye(n_classes)[np.random.choice(n_classes, batch_size)]
            yield image_seqs,labels

#print(u.check_data_dis())
u=utils()
for i in u.batch_dispatch(32):
    print('')