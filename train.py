import tensorflow as tf
import os
from utils import utils
from build_tools import model_tools
from tensorflow.python.client import device_lib
from config import *

print(device_lib.list_local_devices())
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

session = tf.Session()

image_placeholder = tf.placeholder(tf.float32,shape = [None,height,width,color_channels])
output_placeholder = tf.placeholder(tf.float32,shape = [None,1])
tools = utils()

def trainer(network):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=network,labels=output_placeholder)

    loss = tf.reduce_mean(cross_entropy)
    tf.summary.scaler("cost",loss)

    optimizer = tf.train.AdamOptimizer().minimize(loss)

    session.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("tf_summary", graph=tf.get_default_graph())
    merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=4)
    
    for epoch in range(epochs):
        tools.state_restore()
        batch_generator = tools.batch_dispatch(batch_size)

        for ind, batch in enumerate(batch_generator):
            images , ages_prediction = batch
            feed_dict = {image_placeholder:images.T, output_placeholder: ages_prediction}
            _, result = session.run([optimizer,loss],feed_dict=feed_dict)
            print(result)
