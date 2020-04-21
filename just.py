import cv2,os
import numpy as np


data_path = os.path.join('D:','datasets_h5')
valid_folder = os.path.join(data_path,'valid_set')

n=np.load(os.path.join(valid_folder,'0.npz'))
img = n['name1'][0][0]
print(img.shape)
cv2.imshow('image',img)
cv2.waitKey(0)