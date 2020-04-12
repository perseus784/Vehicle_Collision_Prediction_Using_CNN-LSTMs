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
test_folder = os.path.join('datasets','valid_set')

seq_len =15

rough_data = os.path.join('datasets','rough_data')
n_classes = len(os.listdir(rough_data))
width,height,colors = 420,280,3

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




shuffle_organize(rough_data)






