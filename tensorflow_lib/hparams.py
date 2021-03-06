import os
from tensorflow_lib import data_utils


stats_dir = ""
model_dir = ""
model_file_name = ""

inp_dim = 425
out_dim = 187
train_percent,valid_percent,test_percent = 0.7,0.2,0.1
inp_norm = "MINMAX"
out_norm ="MINMAX"

inp_stats_file = os.path.join(stats_dir, "input_%s_%d.norm" %(inp_norm, inp_dim))
out_stats_file = os.path.join(stats_dir, "output_%s_%d.norm" %(out_norm, out_dim))
hidden_layer_type = ['TANH', 'TANH', 'TANH', 'TANH', 'TANH', 'TANH']
hidden_layer_size =[1024, 1024, 1024, 1024, 1024, 1024]

use_high_batch_size = False
sequential_training =True
stateful = False  # RNN
batch_size =  16
seq_length = 200
training_algo = 1
merge_size = 1
bucket_range = 100

encoder_decoder = False
attention = False
cbhg = False

rnn_params = {}
rnn_params['merge_size']   = merge_size
rnn_params['seq_length']   = seq_length
rnn_params['bucket_range'] = bucket_range
rnn_params['stateful']     = stateful
# data
shuffle_data =  True

output_layer_type = 'LINEAR'
loss_function = 'mse'
optimizer ='sgd'

dropout_rate = 0.2
num_of_epochs = 1

json_model_file =os.path.join(model_dir, model_file_name+'.json')
h5_model_file = os.path.join(model_dir, model_file_name+'.h5')