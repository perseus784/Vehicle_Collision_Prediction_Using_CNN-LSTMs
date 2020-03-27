import tensorflow as tf
import os
from utils import utils
from build_tools import model_tools
from tensorflow.python.client import device_lib
from config import *
import model_architecture as model
print(device_lib.list_local_devices())
import statistics
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

session = tf.Session()

image_placeholder = tf.placeholder(tf.float32,shape = [None,height,width,color_channels])
output_placeholder = tf.placeholder(tf.float32,shape = [None,1])
tools = utils()

def percent_error(inp,output):
    s_accuracy = [list((abs(i-j)/i)*100)[0] for i,j in zip(inp,output)]
    accuracy = statistics.mean(s_accuracy)
    return s_accuracy, accuracy

def trainer(network):
    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=network,labels=output_placeholder)
    error = tf.losses.absolute_difference(output_placeholder,network)
    loss = tf.reduce_mean(tf.square(network - output_placeholder))
    #loss = tf.losses.huber_loss(output_placeholder,network)
    tf.summary.scalar("optimizer loss",loss)
    tf.summary.scalar("actual loss", error)


    optimizer = tf.train.AdamOptimizer().minimize(loss)

    session.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("tf_summary", graph=tf.get_default_graph())
    merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=4)
    counter = 0
    for epoch in range(epochs):
        tools.state_restore()
        batch_generator = tools.batch_dispatch(batch_size)
        
        for ind, batch in enumerate(batch_generator):
            images , ages_prediction = batch
            feed_dict = {image_placeholder:images, output_placeholder: ages_prediction}
            _, result,pred,op_error = session.run([optimizer,loss,network,error],feed_dict=feed_dict)
            print(result,pred,ages_prediction,op_error)
            #print(percent_error(ages_prediction,pred))
            counter+=1
            print(counter)
            if ind % 10==0:
                summary = tf.Summary(value=[tf.Summary.Value(tag="actual loss", simple_value=op_error)])
                writer.add_summary(summary,counter)

network = model.create_network(image_placeholder)
trainer(network)