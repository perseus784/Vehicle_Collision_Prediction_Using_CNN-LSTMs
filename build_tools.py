import tensorflow as tf
from config import *

class model_tools:

    def add_weights(self,shape):
        return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.05))

    def add_biases(self,shape):
        return tf.Variable(tf.constant(0.05, shape=shape))

    def conv_layer(self,layer,kernel_size,input_shape,output_shape):
        weights = self.add_weights([kernel_size, kernel_size, kernel_size, input_shape, output_shape])
        biases = self.add_biases([output_shape])
        stride = [1, stride_size, stride_size, stride_size, 1]
        return tf.nn.conv3d(layer, weights, strides=stride, padding='SAME') + biases

    def pooling_layer(self,layer,kernel_size,stride_size):
        kernel = [1, 1, kernel_size, kernel_size, 1]
        stride = [1, 1, stride_size, stride_size, 1]
        return tf.nn.max_pool3d(layer, ksize=kernel, strides=stride, padding='SAME')

    def flattening_layer(self,layer):
        input_size = layer.get_shape().as_list()
        new_size = input_size[-1] * input_size[-2] * input_size[-3]
        return tf.reshape(layer, [input_size[-4], -1,  new_size])

    def fully_connected_layer(self,layer,input_shape,output_shape):
        weights = self.add_weights([input_shape, output_shape])
        biases = self.add_biases([output_shape])
        return tf.matmul(layer,weights) + biases
    
    def dropout_layer(self,layer):
        return tf.nn.dropout(layer,0.3)
    
    def batch_normalization(self,layer):
        return tf.layers.batch_normalization(layer,momentum=0.9)

    def activation(self,layer):
        return tf.nn.relu(layer)

    def op_regression_layer(self,layer):
        return tf.nn.linear(layer)

    def create_lstm_cells(self,number_of_units):
        return tf.contrib.rnn.LSTMCell(number_of_units)
    
    def embedd_inputs(self,embedding_layer,encoder_input):
        encoder_embedded_inputs=tf.nn.embedding_lookup(embedding_layer,encoder_input)
        return encoder_embedded_inputs

    def embed_conv(self,layer):
        input_size = layer.get_shape().as_list()
        new_size = input_size[-1] * input_size[-2] * input_size[-3]
        return tf.reshape(layer, [time, batch_size, new_size])

    def encoder(self,fwd_cell,embedded_inputs):
        encoder_outputs,finalstate = tf.nn.dynamic_rnn(cell=fwd_cell,inputs=embedded_inputs,
                                    dtype=tf.float32,time_major=True)

        encoder_final_state = tf.contrib.rnn.LSTMStateTuple(c=finalstate.c,h=finalstate.h)        

        return encoder_outputs,encoder_final_state
       
    def create_W_B(self,number_of_intents,number_of_units):
        W=tf.Variable(tf.random_uniform([number_of_units,number_of_intents],-0.1,0.1), dtype=tf.float32,name="W_intent")
        b=tf.Variable(tf.zeros([number_of_intents]),dtype=tf.float32,name="b_intent")
        return W,b

    def decoder_intent(self,finalstate_h,w,b):
        logits_intent= tf.matmul(finalstate_h,w)+b
        return logits_intent 
        