import tensorflow as tf
import os
from utils import utils
from build_tools import model_tools
from tensorflow.python.client import device_lib
from config import *
import model_architecture as model
import hybrid_net as hbn
print(device_lib.list_local_devices())
import statistics
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

session = tf.Session()

image_placeholder = tf.placeholder(tf.float32,shape = [batch_size,time, height, width, color_channels])
output_placeholder = tf.placeholder(tf.float32,shape = [batch_size,n_classes])
tools = utils()


def trainer(network):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=network,labels=output_placeholder)
    loss=tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    ac,accuracy = tf.metrics.accuracy(labels = tf.argmax(output_placeholder,1), predictions = tf.argmax(network,1)) 
    tf.summary.scalar("optimizer loss",loss)
    tf.summary.scalar('accuracy',accuracy)
    writer = tf.summary.FileWriter("summary_tf", graph=tf.get_default_graph())
    merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=4)
    counter = 0
    session.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        tools.state_restore()
        batch_generator = tools.batch_dispatch(batch_size)
        
        for ind, batch in enumerate(batch_generator):
            images , class_truth = batch
            feed_dict = {image_placeholder:images, output_placeholder: class_truth}
            _, _loss,pred = session.run([optimizer,loss,network],feed_dict=feed_dict)
            print(_loss,pred,class_truth)
            counter+=1
            print(counter)
            if ind % 10==0:
                summary = tf.Summary(value=[tf.Summary.Value(tag="actual loss", simple_value=_loss)])
                writer.add_summary(summary,counter)

def _trainer(network):
    network.compile(optimizer = 'adam', loss= 'binary_crossentropy',metrics = ['accuracy'])
    batch_generator = tools.batch_dispatch(batch_size)
    network.fit_generator(batch_generator,epochs=epochs,steps_per_epoch=344 // batch_size)

network = hbn.create_network(image_placeholder)
_trainer(network)