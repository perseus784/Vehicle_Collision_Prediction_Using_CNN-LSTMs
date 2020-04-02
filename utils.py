import cv2
import os
import json
import random
from config import *
import numpy as np

dataset='datasets\\train_set\\'
destination='datasets\\fomatted\\train\\'
train_folder = destination

if not os.path.exists(destination):
    os.makedirs(destination)

class utils:

    def __init__(self):
        self.data = 0
        self.data_copy = self.data

    def state_restore(self):
        self.data = self.data_copy

    def create_dataset(self,dataset,destination,dim):
        for i in range(344):
            try:
                if not os.path.exists(destination+'\\{}'.format(i)):
                    os.makedirs(destination+'\\{}'.format(i))
                seq_name = destination+'\\{}'.format(i)
                for j in range(time):
                    im_name = 'set{}_{}.jpeg'.format(i+1,j+1)
                    im=cv2.imread(dataset+im_name)
                    im=cv2.resize(im,(dim,dim))
                    cv2.imwrite(os.path.join(seq_name,str(j)+'.jpg'),im)
            except Exception as e:
                print(e)
                continue

    def get_sequence(self,seq_names):
        seq =[]
        for i in seq_names:
            images = []
            for j in os.listdir(train_folder+i):
                print(os.path.join(train_folder+i,j))
                im=cv2.imread(os.path.join(train_folder+i,j),0)
                images.append(im)
            seq.append(images[:])
        return np.array(seq[:])

    def batch_dispatch(self,batch_size):
        start_index = 0
        end_index = batch_size
        train_data = os.listdir(train_folder)

        while end_index < len(train_data):
            batch_seq = train_data[start_index:end_index]
            print(batch_seq)
            image_seqs = self.get_sequence(batch_seq)
            print('printing here',image_seqs, image_seqs.shape)
            image_seqs = image_seqs.reshape((batch_size,time,height,width,color_channels))
            #print('printing here',image_seqs, image_seqs.shape)
            start_index,end_index = end_index, end_index + batch_size
            labels = np.eye(n_classes)[np.random.choice(n_classes, batch_size)]
            yield[image_seqs,labels]

#print(u.check_data_dis())