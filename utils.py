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

seq_len =15
n_classes = 2
width,height,colors = 420,280,3
one_hot_labels = np.eye(n_classes,dtype='uint8')

class utils:

    def __init__(self):
        pass
            
    def get_sequence(self,seq_names):
        seqs = []
        labels=[]
        for j in seq_names:
            images = np.zeros((seq_len,height,width,colors))
            counter = 0
            class_name = int(j.split('_')[-1])
            for k in os.listdir(os.path.join(train_folder,j)):
                img = cv2.imread(os.path.join(train_folder,j,k))
                images[counter,:,:,:] = img/255
                
                counter+=1
            seqs.append(images)
            labels.append(one_hot_labels[class_name])    
        seqs = np.array(seqs)
        labels = np.array(labels)
        return seqs,labels

    def batch_dispatch(self,batch_size):
        start_index = 0
        end_index = batch_size
        train_data = os.listdir(train_folder)
        random.shuffle(train_data)
        while end_index<len(train_data):
            image_seqs, labels = self.get_sequence(train_data[start_index:end_index])
            #image_seqs = image_seqs.reshape((batch_size,time,height,width,color_channels))
            #labels = np.eye(n_classes)[np.random.choice(n_classes, batch_size)]
            start_index,end_index = end_index,end_index+batch_size
            yield image_seqs,labels


#print(u.check_data_dis())
'''
u=utils()
for i in u.batch_dispatch(32):
    print(' ')'''