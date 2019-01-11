from utils.utils import read_file_list, prepare_file_path_list
from tensorflow_lib import hparams
from utils.utils import read_file_list, prepare_file_path_list


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

    label_normaliser = HTSLabelNormalisation(question_file_name=hparams.question_file_name,
                                             add_frame_features=cfg.add_frame_features,
                                             subphone_feats=hparams.subphone_feats)
    add_feat_dim = sum(cfg.additional_features.values())
    lab_dim = label_normaliser.dimension + add_feat_dim + cfg.appended_input_dim
    if cfg.VoiceConversion:
        lab_dim = cfg.cmp_dim
    logger.info('Input label dimension is %d' % lab_dim)
    suffix = str(lab_dim)