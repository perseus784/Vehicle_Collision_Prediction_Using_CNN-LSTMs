from config import *
import tensorflow as tf
from build_tools import model_tools
sess = tf.Session()

model=model_tools()

def do_conv(input_placeholder):

    network = model.conv_layer(input_placeholder,3,1,16)
    network = model.activation(network)
    network = model.pooling_layer(network,2,2)
    network = model.batch_normalization(network)

    network = model.conv_layer(network,3,16,32)
    network = model.conv_layer(network,3,32,32)
    network = model.activation(network)
    network = model.pooling_layer(network,2,2)
    network = model.batch_normalization(network)

    network = model.conv_layer(network,3,32,64)
    network = model.conv_layer(network,3,64,128)
    network = model.conv_layer(network,3,128,128)
    network = model.activation(network)
    network = model.pooling_layer(network,2,2)
    network = model.batch_normalization(network)

    network = model.conv_layer(network,3,128,256)
    network = model.conv_layer(network,3,256,512)
    network = model.conv_layer(network,3,512,512)
    network = model.activation(network)
    network = model.pooling_layer(network,2,2)
    network = model.batch_normalization(network)

    return network

def create_network(input_data_placeholder):
    
    #embedded
    embedding_layer=tf.Variable(tf.random_uniform([vocabulary_size,embedding_layer_size],-1.0,1.0),dtype=tf.float32,name="embedded_layer")
    conv_op=do_conv(input_data_placeholder)
    
    image_features = model.embed_conv(conv_op)
    
    #creating cells
    encoder_fwd_cell = model.create_lstm_cells(number_of_hiddenunits)

    #encoder
    encoder_outputs,encoder_finalstates=model.encoder(encoder_fwd_cell,image_features)
    print(encoder_outputs,encoder_finalstates)

    #decoder
    w,b=model.create_W_B(number_of_intents,number_of_hiddenunits)
    decoder_network=model.decoder_intent(encoder_finalstates.h,w,b)
    return decoder_network

image_placeholder = tf.placeholder(tf.float32, shape = [batch_size,time,height,width,color_channels])

hyb_network = create_network(image_placeholder)
print(hyb_network)

summary_writer = tf.summary.FileWriter(logdir='summary_tf')
summary_writer.add_graph(graph=tf.get_default_graph())
sess.run(tf.global_variables_initializer())

