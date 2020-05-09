import os,random
import tensorflow as tf
import cv2
import numpy as np
import json
import shutil
from collections import deque
from model_architecture import build_tools
from utils import data_tools
from config import *

tf.reset_default_graph()
tf.set_random_seed(0)
random.seed(0)
np.random.seed(0)

if mode == 'train':
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)
    else:
        shutil.rmtree(model_save_folder)
        os.makedirs(model_save_folder)

    if not os.path.exists(tensorboard_save_folder):
        os.makedirs(tensorboard_save_folder)
    else:
        shutil.rmtree(tensorboard_save_folder)
        os.makedirs(tensorboard_save_folder)


cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1,
                                                 save_weights_only=True,period=4)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_save_folder, histogram_freq=0, write_graph=True,
                                                      write_images=False)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(os.path.join(base_folder,'files','inference_video.avi'),fourcc, 12.0, (width*4,height*4))


def _trainer(network,train_generator,val_generator):
    network.compile(optimizer = 'adam', loss= 'binary_crossentropy',metrics = ['accuracy'])
    network.save_weights(checkpoint_path.format(epoch=0))
    history =network.fit_generator(train_generator,epochs=epochs,steps_per_epoch=len(os.listdir(train_folder)) // batch_size,validation_data=val_generator,validation_steps=1,callbacks=[cp_callback,tensorboard_callback])
    with open(os.path.join(base_folder,'files',model_name,'training_logs.json'),'w') as w:
        json.dump(history.history,w)

def inference(network,video_file):
    print("hgcahjbc")
    image_seq = deque([],8)
    cap = cv2.VideoCapture(video_file)
    counter = 0 
    stat = 'safe'
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            _frame = cv2.resize(frame,(width,height))
            image_seq.append(_frame)
            if counter%2 == 0:
                if len(image_seq)==8:
                    np_image_seqs = np.reshape(np.array(image_seq)/255,(1,time,height,width,color_channels))
                    r = network.predict(np_image_seqs)
                    stat = ['safe', 'collision'][np.argmax(r,1)[0]]
            
            cv2.putText(frame,stat, (230,230), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0),3)
            out.write(frame)
            counter+=1
            print (counter)
        else:
            cap.release()
            out.release()
            cv2.destroyAllWindows()


if  __name__ == "__main__":

    model_tools = build_tools()
    network = model_tools.create_network(model_name)

    if mode == 'train':
        train_generator = data_tools(train_folder,'train')
        valid_generator = data_tools(valid_folder,'valid')
        _trainer(network,train_generator.batch_dispatch(),valid_generator.batch_dispatch())

    else:
        network.load_weights(os.path.join(model_save_folder,'model_weights_032.ckpt'))
        inference(network,os.path.join(base_folder,'files','out1.avi'))

        #testing from batch
        '''test_generator = get_valid_data(test_folder)
        for img_seq,labels in test_generator:
            r = network.predict(img_seq)
            print ('accuracy',np.count_nonzero(np.argmax(r,1)==np.argmax(labels,1))/8)'''