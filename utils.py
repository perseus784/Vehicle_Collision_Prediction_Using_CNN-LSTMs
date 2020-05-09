import cv2
import os
import json
import random
from config import *
import numpy as np
import h5py


train_folder = os.path.join('datasets','train_set')
test_folder = os.path.join('datasets','test_set')
valid_folder = os.path.join('datasets','valid_set')
one_hot_labels = np.eye(n_classes,dtype='uint8')

class data_tools:
    def __init__(self,data_folder,split_name):
        self.data_folder = data_folder
        self._data = os.listdir(self.data_folder)
        if split_name == 'train':
            self.it = int(batch_size/8)
        else:
            self.it = int(32/8)

    def batch_dispatch(self):
        counter = 0
        random.shuffle(self._data)
        while counter<=len(self._data):
            image_seqs=np.empty((0,time,height,width,color_channels))
            labels = np.empty((0,2))
            for i in range(self.it):
                np_data = np.load(os.path.join(self.data_folder,self._data[counter]))
                image_seqs = np.vstack((image_seqs,np_data['name1']/255))
                labels = np.vstack((labels,np_data['name2']))
                counter += 1
                if counter>=len(self._data):
                    counter = 0
                    random.shuffle(self._data)
            yield image_seqs,labels