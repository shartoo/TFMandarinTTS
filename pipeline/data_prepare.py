from utils.utils import read_file_list, prepare_file_path_list
from tensorflow_lib import hparams
from utils.utils import read_file_list, prepare_file_path_list
from acoustic_tools.label_normalisation import HTSLabelNormalisation
from acoustic_tools.silence_remover import SilenceRemover
from acoustic_tools.merge_features import MergeFeat
from acoustic_tools.min_max_norm import MinMaxNormalisation
from acoustic_tools.acoustic_composition import AcousticComposition
from constant import waveform,vocoder_output_dim
from tensorflow_lib import data_utils

import config
import os
import multiprocessing
import numpy as np
from utils import logger

log = logger._set_logger("data_prepare")

def perform_acoustic_composition_on_split(args):
    """ Performs acoustic composition on one chunk of data.
        This is used as input for Pool.map to allow parallel acoustic composition.
    """
    (delta_win, acc_win, in_file_list_dict, nn_cmp_file_list, in_dimension_dict, out_dimension_dict) = args
    acoustic_worker = AcousticComposition(delta_win = delta_win, acc_win = acc_win)
    acoustic_worker.prepare_nn_data(in_file_list_dict, nn_cmp_file_list, in_dimension_dict, out_dimension_dict)

def perform_acoustic_composition(delta_win, acc_win, in_file_list_dict, nn_cmp_file_list, cfg, parallel=True):
    """ Runs acoustic composition from in_file_list_dict to nn_cmp_file_list.
        If parallel is true, splits the data into multiple chunks and calls
        perform_acoustic_composition_on_split for each chunk.
    """
    if parallel:
        num_splits = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(num_splits)

        # split data into a list of num_splits tuples with each tuple representing
        # the parameters for perform_acoustic_compositon_on_split
        splits_full = [
             (delta_win,
              acc_win,
              {stream: in_file_list_dict[stream][i::num_splits] for stream in in_file_list_dict},
              nn_cmp_file_list[i::num_splits],
              cfg.in_dimension_dict,
              cfg.out_dimension_dict
             ) for i in range(num_splits) ]

        pool.map(perform_acoustic_composition_on_split, splits_full)
        pool.close()
        pool.join()
    else:
        acoustic_worker = AcousticComposition(delta_win = delta_win, acc_win = acc_win)
        acoustic_worker.prepare_nn_data(in_file_list_dict, nn_cmp_file_list, cfg.in_dimension_dict, cfg.out_dimension_dict)


def data():
    file_id_list = read_file_list(hparams.file_id_scp)

    data_dir = hparams.data_dir

    inter_data_dir = hparams.inter_data_dir
    nn_cmp_dir = hparams.nn_cmp_dir
    nn_cmp_norm_dir = hparams.nn_cmp_norm_dir
    model_dir = hparams.model_dir
    gen_dir = hparams.gen_dir
    in_file_list_dict = {}
    for feature_name in list(hparams.in_dir_dict.keys()):
        in_file_list_dict[feature_name] = prepare_file_path_list(file_id_list, hparams.in_dir_dict[feature_name], hparams.file_extension_dict[feature_name], False)

    nn_cmp_file_list         = file_paths.get_nn_cmp_file_list()
    nn_cmp_norm_file_list    = file_paths.get_nn_cmp_norm_file_list()

    ###normalisation information
    norm_info_file = file_paths.norm_info_file

    label_normaliser = HTSLabelNormalisation(question_file_name=config.question_file_name,
                                             add_frame_features=config.add_frame_features,
                                             subphone_feats=config.subphone_feats)
    add_feat_dim = sum(config.additional_features.values())
    appended_input_dim = 0
    lab_dim = label_normaliser.dimension + add_feat_dim + appended_input_dim
    log.info('Input label dimension is %d' % lab_dim)
    suffix = str(lab_dim)

    in_label_align_dir = config.in_label_align_dir
    binary_label_dir = config.binary_label_dir
    nn_label_dir = config.nn_label_dir+suffix
    nn_label_norm_dir = config.nn_label_norm_dir+suffix

    in_label_align_file_list = [os.path.join(in_label_align_dir, file_id) for file_id in os.listdir(in_label_align_dir) if file_id.endWiths(".lab")]
    binary_label_file_list = [os.path.join(binary_label_dir,file_id) for file_id in os.listdir(binary_label_dir) if file_id.endswith(".lab")]
    nn_label_file_list = [os.path.join(nn_label_norm_dir,file_id) for file_id in os.listdir(nn_label_norm_dir) if file_id.endswith(".lab")]
    nn_label_norm_file_list = [os.path.join(nn_label_dir,file_id) for file_id in os.listdir(nn_label_dir) if file_id.endswith(".lab")]

    train_file_number = int(len(in_label_align_file_list)*0.7)
    min_max_normaliser = None

    #label_norm_file = label_norm_file
    #test_id_list = test_id_list

    # data normalization

    logger.info('preparing label data (input) using standard HTS style labels')
    label_normaliser.perform_normalisation(in_label_align_file_list, binary_label_file_list, label_type=config.label_type)

    if config.additional_features:
        out_feat_file_list = file_paths.out_feat_file_list
        in_dim = label_normaliser.dimension

        for new_feature, new_feature_dim in config.additional_features.items():
            new_feat_dir = os.path.join(data_dir, new_feature)
            new_feat_file_list = prepare_file_path_list(file_id_list, new_feat_dir, '.' + new_feature)

            merger = MergeFeat(lab_dim=in_dim, feat_dim=new_feature_dim)
            merger.merge_data(binary_label_file_list, new_feat_file_list, out_feat_file_list)
            in_dim += new_feature_dim
            binary_label_file_list = out_feat_file_list
    remover = SilenceRemover(n_cmp=lab_dim, silence_pattern=config.silence_pattern, label_type=config.label_type,
                             remove_frame_features=config.add_frame_features, subphone_feats=config.subphone_feats)
    remover.remove_silence(binary_label_file_list, in_label_align_file_list, nn_label_file_list)
    min_max_normaliser = MinMaxNormalisation(feature_dimension=lab_dim, min_value=0.01, max_value=0.99)
    ###use only training data to find min-max information, then apply on the whole dataset
    min_max_normaliser.find_min_max_values(nn_label_file_list[0:train_file_number])
    ### enforce silence such that the normalization runs without removing silence: only for final synthesis
    min_max_normaliser.normalise_data(nn_label_file_list, nn_label_norm_file_list)
    ## apply minmax normalization
    label_min_vector = min_max_normaliser.min_vector
    label_max_vector = min_max_normaliser.max_vector
    label_norm_info = np.concatenate((label_min_vector, label_max_vector), axis=0)

    label_norm_info = np.array(label_norm_info, 'float32')
    fid = open(label_norm_file, 'wb')
    label_norm_info.tofile(fid)
    fid.close()
    logger.info('saved %s vectors to %s' % (label_min_vector.size, label_norm_file))
    ## make output of duration data
    label_normaliser.prepare_dur_data(in_label_align_file_list, dur_file_list, config.label_type,
                                      "numerical")
    ### make output acoustic data
    if 'dur' in list(config.in_dir_dict.keys()) and config.AcousticModel:
        lf0_file_list = [os.path.join(config.in_lf0_dir,file_id) for file_id in os.listdir(config.in_lf0_dir) if file_id.endswith(".lf0")]
        dur_file_list = [os.path.join(config.in_dur_dir, file_id) for file_id in os.listdir(config.in_dur_dir) if
                         file_id.endswith(".dur")]

        acoustic_worker = AcousticComposition(delta_win=vocoder_output_dim.delta_win, acc_win=vocoder_output_dim.acc_win)
        acoustic_worker.make_equal_frames(dur_file_list, lf0_file_list, cfg.in_dimension_dict)
        acoustic_worker.prepare_nn_data(in_file_list_dict, nn_cmp_file_list, cfg.in_dimension_dict,
                                        cfg.out_dimension_dict)
    else:
        perform_acoustic_composition(vocoder_output_dim.delta_win, vocoder_output_dim.acc_win, in_file_list_dict, nn_cmp_file_list, cfg, parallel=True)

    ## 当前使用HTS标注文件而非 二进制标注
    remover = SilenceRemover(n_cmp=vocoder_output_dim.cmp_dim, silence_pattern=config.silence_pattern, label_type=config.label_type,
                             remove_frame_features=config.add_frame_features, subphone_feats=config.subphone_feats)
    remover.remove_silence(nn_cmp_file_list, in_label_align_file_list, nn_cmp_file_list)  # save to itself

    ### save acoustic normalisation information for normalising the features back
    var_dir = file_paths.var_dir
    var_file_dict = file_paths.get_var_dic()

    ### normalise output acoustic data
    logger.info('normalising acoustic (output) features using method %s' % cfg.output_feature_normalisation)
    cmp_norm_info = None
    # output_feature_normalisation == 'MINMAX':
    min_max_normaliser = MinMaxNormalisation(feature_dimension=vocoder_output_dim.cmp_dim, min_value=0.01, max_value=0.99)
    min_max_normaliser.find_min_max_values(nn_cmp_file_list[0:train_file_number])
    min_max_normaliser.normalise_data(nn_cmp_file_list, nn_cmp_norm_file_list)

    cmp_min_vector = min_max_normaliser.min_vector
    cmp_max_vector = min_max_normaliser.max_vector
    cmp_norm_info = np.concatenate((cmp_min_vector, cmp_max_vector), axis=0)
    # not cfg.GenTestList
    cmp_norm_info = numpy.array(cmp_norm_info, 'float32')
    fid = open(norm_info_file, 'wb')
    cmp_norm_info.tofile(fid)
    fid.close()
    logger.info('saved %s vectors to %s' % (cfg.output_feature_normalisation, norm_info_file))
    feature_index = 0
    for feature_name in list(cfg.out_dimension_dict.keys()):
        feature_std_vector = np.array(
            global_std_vector[:, feature_index:feature_index + cfg.out_dimension_dict[feature_name]], 'float32')
        fid = open(var_file_dict[feature_name], 'w')
        feature_var_vector = feature_std_vector ** 2
        feature_var_vector.tofile(fid)
        fid.close()
        logger.info('saved %s variance vector to %s' % (feature_name, var_file_dict[feature_name]))

        feature_index += cfg.out_dimension_dict[feature_name]
    #

    train_x_file_list, train_y_file_list = file_paths.get_train_list_x_y()
    valid_x_file_list, valid_y_file_list = file_paths.get_valid_list_x_y()
    test_x_file_list, test_y_file_list = file_paths.get_test_list_x_y()

    # we need to know the label dimension before training the DNN
    # computing that requires us to look at the labels
    #
    label_normaliser = HTSLabelNormalisation(question_file_name=config.question_file_name,
                                             add_frame_features=config.add_frame_features,
                                             subphone_feats=config.subphone_feats)
    add_feat_dim = sum(config.additional_features.values())
    lab_dim = label_normaliser.dimension + add_feat_dim #+ config.appended_input_dim

    logger.info('label dimension is %d' % lab_dim)
    hidden_layer_size = hparams.hidden_layer_size
    combined_model_arch = str(len(hidden_layer_size))
    for hid_size in hidden_layer_size:
        combined_model_arch += '_' + str(hid_size)

    nnets_file_name = config.tf_model_dir
    temp_dir_name = file_paths.get_temp_nn_dir_name()
    gen_dir = os.path.join(gen_dir, temp_dir_name)

    ## train model
    tf_instance = TensorflowClass()
    tf_instance.train_tensorflow_model()


def get_x_y(train_id_list,inp_feat_dir,out_feat_dir):
    '''

    :return:
    '''
    inp_train_file_list = data_utils.prepare_file_path_list(train_id_list, inp_feat_dir, ".lab")
    out_train_file_list = data_utils.prepare_file_path_list(train_id_list, out_feat_dir, ".cmp")
    train_x, train_y, train_flen = data_utils.read_data_from_file_list(inp_train_file_list,
                                                                       out_train_file_list,
                                                                       config.inp_dim, config.out_dim,
                                                                       sequential_training=False)
    #### normalize the data ####
    inp_scaler = data_utils.load_norm_stats(config.inp_stats_file, config.inp_dim, method=config.inp_norm)
    out_scaler = data_utils.load_norm_stats(config.out_stats_file, config.out_dim, method=config.out_norm)

    data_utils.norm_data(train_x, inp_scaler,sequential_training=False )
    data_utils.norm_data(train_y, out_scaler,sequential_training=False)

    return train_x,train_y

def get_batch(train_x,train_y,start,batch_size=50):
    utt_keys=train_x.keys()
    if (start+1)*batch_size>len(utt_keys):
        batch_keys=utt_keys[start*batch_size:]
    else:
       batch_keys=utt_keys[start*batch_size:(start+1)*batch_size]
    batch_x_dict=dict([(k,train_x[k]) for k  in batch_keys])
    batch_y_dict=dict([(k,train_y[k]) for k in batch_keys])
    utt_len_batch=[len(batch_x_dict[k])for k in batch_x_dict.keys()]
    return batch_x_dict, batch_y_dict, utt_len_batch


def normlize_data(inp_train_file_list, out_train_file_list):
    ### normalize train data ###
    if os.path.isfile(config.inp_stats_file) and os.path.isfile(config.out_stats_file):
        inp_scaler = data_utils.load_norm_stats(config.inp_stats_file, config.inp_dim, method=config.inp_norm)
        out_scaler = data_utils.load_norm_stats(config.out_stats_file, config.out_dim, method=config.out_norm)
    else:
        print('preparing train_x, train_y from input and output feature files...')
        train_x, train_y, train_flen = data_utils.read_data_from_file_list(inp_train_file_list, out_train_file_list,
            config.inp_dim, config.out_dim, sequential_training=False)

        print('computing norm stats for train_x...')
        inp_scaler = data_utils.compute_norm_stats(train_x, config.inp_stats_file, method=config.inp_norm)

        print('computing norm stats for train_y...')
        out_scaler = data_utils.compute_norm_stats(train_y, config.out_stats_file, method=config.out_norm)
    return inp_scaler,out_scaler


