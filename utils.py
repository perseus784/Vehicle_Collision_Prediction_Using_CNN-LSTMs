import cv2
import os
import json

dataset='datasets\\UTKFace\\'
destination='datasets\\fomatted\\'

if not os.path.exists(destination):
    os.makedirs(destination)

class utils:

    def __init__(self):
        self.data = self.get_data()
        self.data_copy =self.data[:]

    def state_restore(self):
        self.data = self.data_copy

    def create_dataset(self,dataset,destination,dim):
        age_dict={}
        for i,j in enumerate(os.listdir(dataset)):
            age=j.split("_")[0]
            im=cv2.imread(dataset+j,0)
            im=cv2.resize(im,(dim,dim))
            cv2.imwrite(os.path.join(destination,str(i)+'.jpg'),im)
            if str(age) not in age_dict.keys():
                age_dict[str(age)]=[]
            age_dict[str(age)].append(str(i)+'.jpg')

        with open('datasets\\labels.json','w') as fi:
            json.dump(age_dict,fi)

    def get_data(self):
        pass

    def batch_dispatch(self,batch_size):
        start_index = 0
        end_index = batch_size
        while end_index < len(self.data):

            yield[]