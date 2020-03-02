import tensorflow as tf
from config import *
from build_tools import model_tools

model=model_tools()

def create_network(input_placeholder,output_placeholder):

    network = model.conv_layer(input_placeholder,3,1,16)
    network = model.conv_layer(input_placeholder,3,1,16)
    network = model.activation(network)
    network = model.pooling_layer(network,2,2)
    print(network)

    network = model.conv_layer(network,3,16,32)
    network = model.conv_layer(network,3,32,32)
    network = model.activation(network)
    network = model.pooling_layer(network,2,2)
    print(network)

    network = model.conv_layer(network,3,32,64)
    network = model.conv_layer(network,3,64,128)
    network = model.conv_layer(network,3,128,128)
    network = model.activation(network)
    network = model.pooling_layer(network,2,2)
    print(network)

    network = model.conv_layer(network,3,128,256)
    network = model.conv_layer(network,3,256,512)
    network = model.conv_layer(network,3,512,512)
    network = model.activation(network)
    network = model.pooling_layer(network,2,2)
    print(network)

    network,features = model.flattening_layer(network)
    print(network)
    network = model.fully_connected_layer(network,features,1024)
    network = model.activation(network)
    print(network)

    network = model.fully_connected_layer(network,1024,512)
    network = model.activation(network)
    print(network)

    network = model.fully_connected_layer(network,512,1)
    #network = model.op_regression_layer(network)

    print(network)

    return network


images_ph = tf.placeholder(tf.float32, shape=[None, 50, 50, 1])

create_network(images_ph,0)