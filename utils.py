import cv2
import os
import json
import random
from config import *
import numpy as np
dataset='datasets\\UTKFace\\'
destination='datasets\\fomatted\\'

if not os.path.exists(destination):
    os.makedirs(destination)

class utils:

    def __init__(self):

        with open('datasets\\train_labels.json','r') as fi:
            read_data = json.load(fi)
        self.data = read_data
        self.data_copy = self.data

    def state_restore(self):
        self.data = self.data_copy

    def check_data_dis(self):
        with open('datasets\\labels.json','r') as fi:
            read_data = json.load(fi)
        for i in read_data.keys():
            print(len(read_data[i]))


    def create_dataset(self,dataset,destination,dim):
        age_dict={}
        for i,j in enumerate(os.listdir(dataset)):
            age=j[-6:-4]
            im=cv2.imread(dataset+j,0)
            im=cv2.resize(im,(dim,dim))
            cv2.imwrite(os.path.join(destination,str(i)+'.jpg'),im)
            if str(age) not in age_dict.keys():
                age_dict[str(age)]=[]
            age_dict[str(age)].append(str(i)+'.jpg')

        with open('datasets\\labels.json','w') as fi:
            json.dump(age_dict,fi)
    
    def process_dataset(self):
        with open('datasets\\labels.json','r') as fi:
            read_data = json.load(fi)
        l=[]
        for i in read_data:
            for j in read_data[i]:
                l.append([i,j])

        _testlen=int(len(l)//10)
        random.shuffle(l)
        train_set = l[:len(l)-_testlen]
        test_set = l[len(train_set):]
        print(len(train_set),len(test_set))
        with open('datasets\\train_labels.json','w') as fi:
            json.dump(train_set,fi)
        with open('datasets\\test_labels.json','w') as fi:
            json.dump(test_set,fi)

    def get_images(self,image_names):
        all_images=[]
        for i in image_names:
            img = cv2.imread(os.path.join(image_path,i),0)
            img = img.reshape(50,50,1)
            all_images.append(img)

        return all_images

    def batch_dispatch(self,batch_size):
        start_index = 0
        end_index = batch_size
        while end_index < len(self.data):
            data = self.data[start_index:end_index]
            ages,im_names = zip(*data)
            images = self.get_images(im_names)
            ages = np.array(list(map(int, ages))).reshape(batch_size,1)
            start_index,end_index = end_index, end_index + batch_size
            yield[np.array(images),np.array(ages)]


#u=utils()
#print(u.check_data_dis())