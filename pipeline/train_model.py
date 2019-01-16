from pipeline import data_prepare
from tensorflow_lib import data_utils,hparams
from tensorflow.contrib.layers import dropout,fully_connected,batch_norm

import config
from utils import logger
import tensorflow as tf
import numpy as np
import os

logger = logger._set_logger("train_model")

inp_train_file_list, out_train_file_list =[] ,[]
inp_scaler,out_scaler = data_prepare.normlize_data(inp_train_file_list, out_train_file_list)
train_x, train_y, train_flen = data_utils.read_data_from_file_list(inp_train_file_list, out_train_file_list,
                                                                   config.inp_dim, config.out_dim,
                                                                   sequential_training=False)
#### normalize the data ####
data_utils.norm_data(train_x, inp_scaler,sequential_training=False)
data_utils.norm_data(train_y, out_scaler,sequential_training=False)

def feedforward_model():
    layer_list = []
    graph = tf.Graph()
    with graph.as_default() as g:
        is_training_batch = tf.placeholder(tf.bool, shape=(), name="is_training_batch")
        bn_params = {"is_training": is_training_batch, "decay": 0.99, "updates_collections": None}
        g.add_to_collection("is_training_batch", is_training_batch)
        with tf.name_scope("input"):
            input_layer = tf.placeholder(dtype=tf.float32, shape=(None, config.inp_dim), name="input_layer")
            if hparams.dropout_rate != 0.0:
                print("Using dropout to avoid overfitting and the dropout rate is", hparams.dropout_rate)
                is_training_drop = tf.placeholder(dtype=tf.bool, shape=(), name="is_training_drop")
                input_layer_drop = dropout(input_layer, hparams.dropout_rate, is_training=is_training_drop)
                layer_list.append(input_layer_drop)
                g.add_to_collection(name="is_training_drop", value=is_training_drop)
            else:
                layer_list.append(input_layer)
        g.add_to_collection("input_layer", layer_list[0])
        for i in range(len(hparams.hidden_layer_size)):
            with tf.name_scope("hidden_layer_" + str(i + 1)):
                if hparams.dropout_rate != 0.0:
                    last_layer = layer_list[-1]
                    if hparams.hidden_layer_type[i] == "tanh":
                        new_layer = fully_connected(last_layer, hparams.hidden_layer_size[i], activation_fn=tf.nn.tanh,
                                                    normalizer_fn=batch_norm,
                                                    normalizer_params=bn_params)
                    if hparams.hidden_layer_type[i] == "sigmoid":
                        new_layer = fully_connected(last_layer, hparams.hidden_layer_size[i], activation_fn=tf.nn.sigmoid,
                                                    normalizer_fn=batch_norm,
                                                    normalizer_params=bn_params)
                    new_layer_drop = dropout(new_layer, hparams.dropout_rate, is_training=is_training_drop)
                    layer_list.append(new_layer_drop)
                else:
                    last_layer = layer_list[-1]
                    if config.hidden_layer_type[i] == "tanh":
                        new_layer = fully_connected(last_layer, hparams.hidden_layer_size[i], activation_fn=tf.nn.tanh,
                                                    normalizer_fn=batch_norm,
                                                    normalizer_params=bn_params)
                    if hparams.hidden_layer_type[i] == "sigmoid":
                        new_layer = fully_connected(last_layer, hparams.hidden_layer_size[i], activation_fn=tf.nn.sigmoid,
                                                    normalizer_fn=batch_norm,
                                                    normalizer_params=bn_params)
                    layer_list.append(new_layer)
        with tf.name_scope("output_layer"):
            output_layer = None
            if hparams.output_layer_type == "linear":
                output_layer = fully_connected(layer_list[-1], config.out_dim, activation_fn=None)
            elif hparams.output_layer_type == "tanh":
                output_layer = fully_connected(layer_list[-1], config.out_dim, activation_fn=tf.nn.tanh)
            g.add_to_collection(name="output_layer", value=output_layer)
    return graph


def train_feedforward_model(ckpt_dir,train_x, train_y, batch_size=256, num_of_epochs=10, shuffle_data=True):
        seed=12345
        np.random.seed(seed)
        print(train_x.shape)
        training_op = tf.train.AdamOptimizer()
        with tf.Graph.as_default() as g:
           output_data=tf.placeholder(dtype=tf.float32,shape=(None,config.out_dim),name="output_data")
           input_layer=g.get_collection(name="input_layer")[0]
           is_training_batch=g.get_collection(name="is_training_batch")[0]
           is_training_drop = True
           if hparams.dropout_rate!=0.0:
              is_training_drop=g.get_collection(name="is_training_drop")[0]
           with tf.name_scope("loss"):
               output_layer=g.get_collection(name="output_layer")[0]
               loss=tf.reduce_mean(tf.square(output_layer-output_data),name="loss")
           with tf.name_scope("train"):
                training_op=training_op.minimize(loss)
           init=tf.global_variables_initializer()
           saver=tf.train.Saver()
           with tf.Session() as sess:
             init.run();
             summary_writer=tf.summary.FileWriter(os.path.join(ckpt_dir,"losslog"),sess.graph)
             for epoch in range(num_of_epochs):
                 L=1;overall_loss=0
                 for iteration in range(int(train_x.shape[0]/batch_size)+1):
                    if (iteration+1)*batch_size>train_x.shape[0]:
                        x_batch,y_batch=train_x[iteration*batch_size:],train_y[iteration*batch_size:]
                        if x_batch!=[]:
                           L+=1
                        else:continue
                    else:
                        x_batch,y_batch=train_x[iteration*batch_size:(iteration+1)*batch_size,], train_y[iteration*batch_size:(iteration+1)*batch_size]
                        L+=1
                    if hparams.dropout_rate!=0.0:
                       _,batch_loss=sess.run([training_op,loss],feed_dict={input_layer:x_batch,output_data:y_batch,is_training_drop:True,is_training_batch:True})
                    else:
                       _,batch_loss=sess.run([training_op,loss],feed_dict={input_layer:x_batch,output_data:y_batch,is_training_batch:True})
                    overall_loss+=batch_loss
                 print("Epoch ",epoch+1, "Finishes","Training loss:",overall_loss/L)
             saver.save(sess,os.path.join(ckpt_dir,"mandarin_tts.ckpt"))
             print("The model parameters are saved")


