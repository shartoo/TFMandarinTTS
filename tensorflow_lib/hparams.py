import os
from tensorflow_lib import data_utils


stats_dir = ""
model_dir = ""
model_file_name = ""

inp_dim = 425
out_dim = 187

inp_norm = "MINMAX"
out_norm ="MINMAX"

inp_stats_file = os.path.join(stats_dir, "input_%d_%s_%d.norm" %(int(train_file_number), inp_norm, inp_dim))
out_stats_file = os.path.join(stats_dir, "output_%d_%s_%d.norm" %(int(train_file_number), out_norm, out_dim))


inp_scaler =  data_utils.load_norm_stats(inp_stats_file, inp_dim, method=inp_norm)
out_scaler = data_utils.load_norm_stats(out_stats_file, out_dim, method=out_norm)

hidden_layer_type = ['TANH', 'TANH', 'TANH', 'TANH', 'TANH', 'TANH']
hidden_layer_size =[1024, 1024, 1024, 1024, 1024, 1024]

#  RNN
('model_file_name', 'feed_forward_6_tanh', 'Architecture', 'model_file_name'),
('stateful', False, 'Architecture', 'stateful'),
('use_high_batch_size', False, 'Architecture', 'use_high_batch_size'),

('training_algo', 1, 'Architecture', 'training_algo'),
('merge_size', 1, 'Architecture', 'merge_size'),
('seq_length', 200, 'Architecture', 'seq_length'),
('bucket_range', 100, 'Architecture', 'bucket_range'),

('encoder_decoder', False, 'Architecture', 'encoder_decoder'),
('attention', False, 'Architecture', 'attention'),
("cbhg", False, "Architecture", "cbhg"),

use_high_batch_size = False
sequential_training =True
stateful = False  # RNN
batch_size =  16
seq_length = 200
training_algo = 1

merge_size = 1
bucket_range = 100
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


dropout_rate = 0.0
num_of_epochs = 1

json_model_file =os.path.join(model_dir, model_file_name+'.json')
h5_model_file = os.path.join(model_dir, model_file_name+'.h5')