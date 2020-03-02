import cv2
import os
import json
dataset='datasets\\UTKFace\\'
destination='datasets\\fomatted\\'
if not os.path.exists(destination):
    os.makedirs(destination)

class utils:

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


    def batch_dispatch(self):
        return 0