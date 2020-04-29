# -*- coding: utf-8 -*-

import os,random
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import cv2
import numpy as np
import json
import shutil
from collections import deque

tf.reset_default_graph()
tf.set_random_seed(0)
random.seed(0)
np.random.seed(0)

#config
base_folder = os.path.abspath(os.curdir)
data_path = os.path.join(base_folder,'datasets_h5')
train_folder = os.path.join(data_path,'train_set')
test_folder = os.path.join(data_path,'test_set')
valid_folder = os.path.join(data_path,'valid_set')
model_name = 'vgg'
model_save_folder = os.path.join(base_folder,'files',model_name,'model_folder')
tensorboard_save_folder = os.path.join(base_folder,'files',model_name,'tensorboard_folder')
'''
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
'''
epochs = 50
time = 8
n_classes = 2
width,height,color_channels = 210,140,3
number_of_hiddenunits = 32
batch_size = 16
checkpoint_path = os.path.join(model_save_folder,"model_weights_{epoch:03d}.ckpt")
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1,
                                                 save_weights_only=True,period=4)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_save_folder, histogram_freq=0, write_graph=True,
                                                      write_images=False)
#update_freq='batch',profile_batch=2,embeddings_freq=0,embeddings_metadata=None
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('out.avi',fourcc, 12.0, (210*4,140*4))

def batch_dispatch(data_folder):
    _data = os.listdir(data_folder)
    counter = 0
    it = int(batch_size/8)

    while counter<=len(_data):

        image_seqs=np.empty((0,time,height,width,color_channels))
        labels = np.empty((0,2))
        for i in range(it):
            np_data = np.load(os.path.join(data_folder,_data[counter]))
            image_seqs = np.vstack((image_seqs,np_data['name1']/255))
            labels = np.vstack((labels,np_data['name2']))
            '''np_data = np.load(os.path.join(data_folder,_data[counter]))
            image_seqs = np_data['name1']/255
            labels = np_data['name2']'''
            counter += 1
        if counter<=len(_data):
            counter = 0
        random.shuffle(_data)
        yield image_seqs,labels

def get_valid_data(data_folder):
    _data = os.listdir(data_folder)
    random.shuffle(_data)
    counter = 0
    it = 1 #int(batch_size/8)

    while counter<=len(_data):
        image_seqs=np.empty((0,time,height,width,color_channels))
        labels = np.empty((0,2))
        for i in range(it):
            print(counter)
            np_data = np.load(os.path.join(data_folder,_data[counter]))
            image_seqs = np.vstack((image_seqs,np_data['name1']/255))
            labels = np.vstack((labels,np_data['name2']))
            '''np_data = np.load(os.path.join(data_folder,_data[counter]))
            image_seqs = np_data['name1']/255
            labels = np_data['name2']'''
            counter += 1
        
        if counter>=len(_data):
            counter = 0
        yield image_seqs,labels


class build_model:
    def __init__(self):
        pass

    def inception_module(self,conv_model,filter1,filter2_1,filter2_2,filter3_1,filter3_2,filter4):
        conv_1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filter1, (1,1), padding='same', activation='relu'))(conv_model)

        conv_3_1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filter2_1, (1,1), padding='same', activation='relu'))(conv_model)
        conv_3_2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filter2_2, (3,3), padding='same', activation='relu'))(conv_3_1)

        conv_5_1 =  tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filter3_1, (1,1), padding='same', activation='relu'))(conv_model)
        conv_5_2 =  tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filter3_2, (5,5), padding='same', activation='relu'))(conv_5_1) 

        pooling_layer =  tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(3,3), padding='SAME',strides=(1,1)))(conv_model)
        conv_pooling =  tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filter4, (1,1), padding='same', activation='relu'))(pooling_layer) 

        conv_model = tf.keras.layers.Concatenate(axis= 4)([conv_1,conv_3_2,conv_5_2,conv_pooling])

        return conv_model

    def resnet_module(self):
        pass

    def get_conv_vgg(self,input_batch):

        '''conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'))(input_batch)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(conv_model)'''

        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu') )(input_batch)
        #conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu') )(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='SAME',strides=(2,2)))(conv_model)

        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu') )(conv_model)
        #conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu') )(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='SAME',strides=(2,2)))(conv_model)

        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu') )(conv_model)
        #conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu') )(conv_model)
        #conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu') )(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='SAME',strides=(2,2)))(conv_model)

        '''conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu') )(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu') )(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu') )(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='SAME',strides=(2,2)))(conv_model)'''

        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu') )(conv_model)
        #conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu') )(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu') )(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='SAME',strides=(2,2)))(conv_model)

        #embedded
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(conv_model)
        '''conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024,activation='relu'))(conv_model)'''
        #conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(conv_model)
        #conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.5))(conv_model)
        '''conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024,activation='relu'))(conv_model)'''

        return conv_model

    def get_conv_inception(self,input_batch):

        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (7,7), padding='same', activation='relu') )(input_batch)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(3,3), padding='SAME',strides=(2,2)))(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (1,1), padding='same', activation='relu') )(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(192, (3,3), padding='same', activation='relu') )(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(3,3), padding='SAME',strides=(2,2)))(conv_model) 
        conv_model = self.inception_module(conv_model,64,96,128,16,32,32)
        conv_model = self.inception_module(conv_model,128,128,192,32,96,64)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(3,3), padding='SAME',strides=(2,2)))(conv_model) 
        conv_model = self.inception_module(conv_model,192,96,208,16,48,64)
        conv_model = self.inception_module(conv_model,160,112,224,24,64,64)
        '''conv_model = self.inception_module(conv_model,128,128,256,24,64,64)
        conv_model = self.inception_module(conv_model,112,144,288,32,64,64)
        conv_model = self.inception_module(conv_model,256,160,320,32,128,128)'''
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(3,3), padding='SAME',strides=(2,2)))(conv_model) 
        conv_model = self.inception_module(conv_model,256,160,320,32,128,128)
        #conv_model = self.inception_module(conv_model,384,192,384,48,128,128)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.AveragePooling2D(pool_size=(7,7), padding='SAME',strides=(1,1)))(conv_model) 
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.5))(conv_model) 
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024,activation='relu'))(conv_model) 
        return conv_model

    def get_conv_resnet(self,input_batch):
        return conv_model

    def get_conv_inception_resnet(self,conv_model):
        return conv_model

def create_network(model_tools):
    input_batch = tf.keras.layers.Input(shape = (time,height,width,color_channels))

    if model_name == 'vgg':
        image_features = model_tools.get_conv_vgg(input_batch)

    elif model_name == 'inception':
        image_features = model_tools.get_conv_inception(input_batch)

    lstm_network = tf.keras.layers.LSTM(number_of_hiddenunits, return_sequences=False,dropout=0.5,recurrent_dropout=0.5)(image_features)
    #lstm_network = tf.keras.layers.LSTM(number_of_hiddenunits, return_sequences=True,dropout=0.5,recurrent_dropout=0.5)(lstm_network)
    #lstm_network = tf.keras.layers.LSTM(number_of_hiddenunits, return_sequences=False,dropout=0.5,recurrent_dropout=0.5)(lstm_network)
    lstm_network = tf.keras.layers.Dense(1024,activation='relu')(lstm_network)
    lstm_network = tf.keras.layers.BatchNormalization()(lstm_network)
    lstm_network = tf.keras.layers.Dropout(0.5)(lstm_network)
    lstm_network = tf.keras.layers.Dense(512,activation='relu')(lstm_network)
    lstm_network = tf.keras.layers.Dropout(0.5)(lstm_network)
    lstm_network = tf.keras.layers.Dense(64,activation='relu')(lstm_network)
    lstm_network = tf.keras.layers.Dropout(0.5)(lstm_network)    
    lstm_network = tf.keras.layers.Dense(n_classes,activation='softmax')(lstm_network)
    full_network = tf.keras.Model([input_batch],lstm_network)
    full_network.summary()
    return full_network
    
def _trainer(network):
    network.compile(optimizer = 'adam', loss= 'binary_crossentropy',metrics = ['accuracy'])
    batch_generator = batch_dispatch(train_folder)
    val_batch = get_valid_data(valid_folder)
    network.save_weights(checkpoint_path.format(epoch=0))
    history =network.fit_generator(batch_generator,epochs=epochs,steps_per_epoch=len(os.listdir(train_folder)) // batch_size,validation_data=val_batch,validation_steps=1,callbacks=[cp_callback,tensorboard_callback])
    with open(os.path.join(base_folder,'files',model_name,'training_logs.json'),'w') as w:
        json.dump(history.history,w)


def inference(network,video_file):
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
    model_tools = build_model()
    network = create_network(model_tools)
    network.load_weights(os.path.join(model_save_folder,'model_weights_032.ckpt'))
    inference(network,os.path.join(base_folder,'files','output.avi'))
    '''test_generator = get_valid_data(test_folder)
    for img_seq,labels in test_generator:
        r = network.predict(img_seq)
        print ('accuracy',np.count_nonzero(np.argmax(r,1)==np.argmax(labels,1))/8)'''
    #_trainer(network)
