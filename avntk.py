# -*- coding: utf-8 -*-

import os,random
import tensorflow as tf
import cv2
import numpy as np


#config
base_folder = os.path.abspath(os.curdir)
data_path = os.path.join(base_folder,'datasets')
train_folder = os.path.join(data_path,'train_set')
test_folder = os.path.join(data_path,'test_set')
valid_folder = os.path.join(data_path,'valid_set')
model_save_folder = os.path.join(base_folder,'files','model_folder')
tensorboard_save_folder = os.path.join(base_folder,'files','tensorboard_folder')


if not os.path.exists(model_save_folder):
    os.makedirs(model_save_folder)
if not os.path.exists(tensorboard_save_folder):
    os.makedirs(tensorboard_save_folder)

epochs = 50
time = 8
n_classes = 2
width,height,color_channels = 210,140,3
number_of_hiddenunits = 20
batch_size = 16
checkpoint_path = "model_weights_{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1,
                                                 save_weights_only=True,save_freq=1)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_save_folder, histogram_freq=0, write_graph=True,
                                                      write_images=False,update_freq='batch', profile_batch=2,
                                                      embeddings_freq=0,embeddings_metadata=None)

def batch_dispatch(data_folder):
    _data = os.listdir(data_folder)
    random.shuffle(_data)
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
        yield image_seqs,labels

def get_valid_data(data_folder):
    _data = os.listdir(data_folder)
    random.shuffle(_data)
    image_seqs=np.empty((0,time,height,width,color_channels))
    labels = np.empty((0,2))
    for i in range(len(_data)):
      np_data = np.load(os.path.join(data_folder,_data[i]))
      image_seqs = np.vstack((image_seqs,np_data['name1']))
      labels = np.vstack((labels,np_data['name2']))
    return image_seqs/255,labels
def create_network():
    conv_model = tf.keras.models.Sequential()
    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(16, (3,3), padding='same', activation='relu'),input_shape =(time,height,width,color_channels) ))
    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()))
    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='SAME')))

    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu') ))
    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu') ))
    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()))
    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='SAME')))   
    
    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu') ))
    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu') ))
    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()))
    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='SAME'))) 

    '''conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu') ))
    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu') ))
    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()))
    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='SAME'))) 

    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu') ))
    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu') ))
    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()))
    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='SAME')))'''

    #embedded
    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2048)))
    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024)))

    #image_features = model.embed_conv(conv_model)
    conv_model.add(tf.keras.layers.LSTM(number_of_hiddenunits, return_sequences=False))
    conv_model.add(tf.keras.layers.Dense(16))
    conv_model.add(tf.keras.layers.Dense(2))
    conv_model.add(tf.keras.layers.Activation('softmax'))
    conv_model.summary()
    return conv_model
    
def _trainer(network):
    network.compile(optimizer = 'adam', loss= 'binary_crossentropy',metrics = ['accuracy'])
    batch_generator = batch_dispatch(train_folder)
    val_batch = get_valid_data(valid_folder)
    gpu_info = !nvidia-smi
    gpu_info = '\n'.join(gpu_info)
    if gpu_info.find('failed') >= 0:
      print('Select the Runtime → "Change runtime type" menu to enable a GPU accelerator, ')
      print('and then re-execute this cell.')
    else:
      print(gpu_info)
    ram_gb = virtual_memory().total / 1e9
    print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))
    network.save_weights(checkpoint_path.format(epoch=0))
    network.fit(batch_generator,epochs=epochs,steps_per_epoch=len(os.listdir(train_folder)) // batch_size,
                validation_data=val_batch,validation_steps=1,callbacks=[cp_callback,tensorboard_callback])

network = create_network()
_trainer(network)