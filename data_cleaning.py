import cv2
import os
import json
import random
from config import *
import numpy as np
import h5py
from distutils.dir_util import copy_tree


alldata_folder = os.path.join('datasets','formatted')

train_folder = os.path.join('datasets','train_set')
test_folder = os.path.join('datasets','test_set')
valid_folder = os.path.join('datasets','valid_set')

seq_len = 8

'''rough_data = os.path.join('datasets','rough_data')
n_classes = len(os.listdir(rough_data))'''
n_classes = 2
width,height,colors = 210,140,3
one_hot_labels = np.eye(n_classes,dtype='uint8')

def shuffle_organize(rough_data_folder):
    n_list = []
    n_samples = 10e10
    rough_data_classes = os.listdir(rough_data_folder)
    for i in rough_data_classes:
        files_in_class = os.listdir(os.path.join(rough_data_folder,i))
        if n_samples > len(files_in_class):
            n_samples = len(files_in_class)
        random.shuffle(files_in_class)
        n_list.append(files_in_class)
    shuffled_list = []
    for i in n_list:
        shuffled_list.append(random.sample(i,n_samples))
    counter = 0
    for class_number,sequences_list in enumerate(shuffled_list):
        for i in sequences_list:
            src_folder = os.path.join(rough_data_folder,rough_data_classes[class_number],i)
            dst_folder = os.path.join(alldata_folder,'{}_{}'.format(str(counter),str(class_number)))
            counter += 1
            copy_tree(src_folder, dst_folder)
            print('sequences complete',counter)

def assort():
    all_data = os.listdir(alldata_folder)
    random.shuffle(all_data)
    tn = len(all_data)
    train_p = int(tn*0.75)
    test_p = int(tn*0.15)

    train_set = all_data[:train_p]
    test_set = all_data[train_p:train_p+test_p]
    val_set = all_data[train_p+test_p:]


def get_sequence(seq_names,src_folder):
    seqs = []
    labels=[]
    for j in seq_names:
        images = np.zeros((seq_len,height,width,colors))
        counter = 0
        class_name = int(j.split('_')[-1])
        folder_of_seq = os.listdir(os.path.join(src_folder,j))
        lsorted = sorted(folder_of_seq,key=lambda x: int(os.path.splitext(x)[0]))
        lsorted = lsorted[0::2]
        for k in lsorted:
            img = cv2.imread(os.path.join(src_folder,j,k))
            images[counter,:,:,:] = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
            counter+=1
        seqs.append(images)
        labels.append(one_hot_labels[class_name])    
    seqs = np.array(seqs,dtype='uint8')
    labels = np.array(labels,dtype='uint8')
    return seqs,labels

def batch_dispatch(batch_size,src_folder,dst_folder):
    start_index = 0
    end_index = batch_size
    _data = os.listdir(src_folder)
    random.shuffle(_data)
    counter = len(os.listdir(dst_folder))
    while end_index<=len(_data):
        image_seqs, labels = get_sequence(_data[start_index:end_index],src_folder)
        #image_seqs = image_seqs.reshape((batch_size,time,height,width,color_channels))
        #labels = np.eye(n_classes)[np.random.choice(n_classes, batch_size)]
        np.savez(os.path.join(dst_folder,'{}.npz'.format(str(counter))),name1 = image_seqs,name2 = labels)
        '''with h5py.File(os.path.join(dst_folder,'{}.hdf5'.format(str(counter))), 'w') as f:
            f.create_dataset('sequences', data=image_seqs)
            f.create_dataset('labels', data=labels)        '''
        start_index,end_index = end_index,end_index+batch_size
        counter += 1
        print(counter)

dst_folder = os.path.join('D:','datasets_h5','train_set')
batch_dispatch(8,valid_folder,dst_folder)





