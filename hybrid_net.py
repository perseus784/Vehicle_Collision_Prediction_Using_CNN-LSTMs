from config import *
import tensorflow as tf
from build_tools import model_tools
#sess = tf.Session()

model=model_tools()

def do_conv(input_placeholder):

    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(16, (3,3), padding='same', activation='relu'),input_shape =(time,height,width,color_channels) ))
    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='SAME')))
    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()))

    #network = model.activation(network)
    '''network = model.pooling_layer(network,2,2)
    network = model.batch_normalization(network)'''

    '''
    network = model.conv_layer(network,3,16,32,1)
    network = model.conv_layer(network,3,32,32,1)
    #network = model.activation(network)
    network = model.pooling_layer(network,2,2)
    network = model.batch_normalization(network)

    network = model.conv_layer(network,3,32,64,1)
    network = model.conv_layer(network,3,64,128,1)
    network = model.conv_layer(network,3,128,128,1)
    #network = model.activation(network)
    network = model.pooling_layer(network,2,2)
    network = model.batch_normalization(network)

    network = model.conv_layer(network,3,128,256,1)
    network = model.conv_layer(network,3,256,512,1)
    network = model.conv_layer(network,3,512,512,1)
    #network = model.activation(network)
    network = model.pooling_layer(network,2,2)
    network = model.batch_normalization(network)'''

    return conv_model

def create_network(input_data_placeholder):
    conv_model = tf.keras.models.Sequential()

    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(16, (3,3), padding='same', activation='relu'),input_shape =(time,height,width,color_channels) ))
    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='SAME')))
    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()))
    #embedded

    conv_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
    #image_features = model.embed_conv(conv_model)

    conv_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(number_of_hiddenunits, return_sequences=False)))
    conv_model.add(tf.keras.layers.Dense(16))
    conv_model.add(tf.keras.layers.Dense(2))
    conv_model.add(tf.keras.layers.Activation('softmax'))
    conv_model.summary()

    return conv_model

    '''
    #creating cells
    encoder_fwd_cell = model.create_lstm_cells(number_of_hiddenunits)

    #encoder
    encoder_outputs,encoder_finalstates=model.encoder(encoder_fwd_cell,image_features)

    #decoder
    w,b=model.create_W_B(number_of_intents,number_of_hiddenunits)
    decoder_network=model.decoder_intent(encoder_finalstates.h,w,b)
    return decoder_network'''

'''
image_placeholder = tf.placeholder(tf.float32, shape = [None,time,height,width,color_channels])

hyb_network = create_network(image_placeholder)
print(hyb_network)'''

'''
summary_writer = tf.summary.FileWriter(logdir='summary_tf')
summary_writer.add_graph(graph=tf.get_default_graph())
sess.run(tf.global_variables_initializer())'''

