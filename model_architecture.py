from config import *
import tensorflow as tf

class build_tools:

    def __init__(self):
        pass

    def inception_module(self,conv_model,filter1,filter2_1,filter2_2,filter3_1,filter3_2,filter4):
        conv_1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filter1, (1,1), padding='same', activation='relu'))(conv_model)
        conv_1 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(conv_1)

        conv_3_1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filter2_1, (1,1), padding='same', activation='relu'))(conv_model)
        conv_3_2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filter2_2, (3,3), padding='same', activation='relu'))(conv_3_1)
        conv_3_2 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(conv_3_2)

        conv_5_1 =  tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filter3_1, (1,1), padding='same', activation='relu'))(conv_model)
        conv_5_2 =  tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filter3_2, (5,5), padding='same', activation='relu'))(conv_5_1) 
        conv_5_2 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(conv_5_2)

        pooling_layer =  tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(3,3), padding='SAME',strides=(1,1)))(conv_model)
        conv_pooling =  tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filter4, (1,1), padding='same', activation='relu'))(pooling_layer) 
        conv_pooling = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(conv_pooling)

        conv_model = tf.keras.layers.Concatenate(axis= 4)([conv_1,conv_3_2,conv_5_2,conv_pooling])

        return conv_model

    def get_conv_vgg(self,input_batch):

        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu') )(input_batch)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='SAME',strides=(2,2)))(conv_model)

        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu') )(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='SAME',strides=(2,2)))(conv_model)

        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu') )(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu') )(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='SAME',strides=(2,2)))(conv_model)

        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu') )(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu') )(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu') )(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='SAME',strides=(2,2)))(conv_model)

        #embedded
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(conv_model)

        return conv_model

    def get_conv_inception(self,input_batch):

        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (7,7), padding='same', activation='relu') )(input_batch)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(3,3), padding='SAME',strides=(2,2)))(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(192, (3,3), padding='same', activation='relu') )(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(conv_model)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(3,3), padding='SAME',strides=(2,2)))(conv_model) 

        conv_model = self.inception_module(conv_model,128,128,192,32,96,64)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(3,3), padding='SAME',strides=(2,2)))(conv_model) 

        conv_model = self.inception_module(conv_model,256,160,320,32,128,128)
        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(3,3), padding='SAME',strides=(2,2)))(conv_model)

        conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(conv_model)
        return conv_model

    def create_network(self,model_name):
        input_batch = tf.keras.layers.Input(shape = (time,height,width,color_channels))

        if model_name == 'vgg':
            image_features = self.get_conv_vgg(input_batch)
            lstm_network = tf.keras.layers.LSTM(number_of_hiddenunits, return_sequences=True,dropout=0.5,recurrent_dropout=0.5)(image_features)
            lstm_network = tf.keras.layers.LSTM(number_of_hiddenunits, return_sequences=False,dropout=0.5,recurrent_dropout=0.5)(lstm_network)
            lstm_network = tf.keras.layers.Dense(1024,activation='relu')(lstm_network)
            lstm_network = tf.keras.layers.BatchNormalization()(lstm_network)
            lstm_network = tf.keras.layers.Dropout(0.5)(lstm_network)
            lstm_network = tf.keras.layers.Dense(512,activation='relu')(lstm_network)
            lstm_network = tf.keras.layers.Dropout(0.5)(lstm_network)
            lstm_network = tf.keras.layers.Dense(64,activation='relu')(lstm_network)
            lstm_network = tf.keras.layers.Dropout(0.5)(lstm_network)    
            lstm_network = tf.keras.layers.Dense(n_classes,activation='softmax')(lstm_network)

        elif model_name == 'inception':
            image_features = self.get_conv_inception(input_batch)
            lstm_network = tf.keras.layers.LSTM(number_of_hiddenunits, return_sequences=True,dropout=0.5,recurrent_dropout=0.5)(image_features)
            lstm_network = tf.keras.layers.LSTM(number_of_hiddenunits, return_sequences=False,dropout=0.5,recurrent_dropout=0.5)(lstm_network)
            lstm_network = tf.keras.layers.Dense(512,activation='relu')(lstm_network)
            lstm_network = tf.keras.layers.Dense(64,activation='relu')(lstm_network)
            lstm_network = tf.keras.layers.Dropout(0.5)(lstm_network)    
            lstm_network = tf.keras.layers.Dense(n_classes,activation='softmax')(lstm_network)

        full_network = tf.keras.Model([input_batch],lstm_network)
        full_network.summary()
        return full_network
