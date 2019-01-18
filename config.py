import os
import configparser
from constant import vocoder_output_dim,waveform
from utils import logger

print(os.listdir("../data"))
log = logger.get_logger("config_log","../data/log/mandarin_tts.log")

cf = configparser.ConfigParser()
pwd = os.getcwd()
print(os.listdir("../resources/mandarin"))
cf.read(os.path.join(pwd,'../resources/cfgs','cfg.config'))
#
add_frame_features = ""
inp_dim = 425
out_dim = 187
inp_norm = "MINMAX"
out_norm = "MINMAX"
output_features = ['mgc','lf0', 'vuv', 'bap']
gen_wav_features = ['mgc', 'bap', 'lf0']

print(cf.sections())
### acoustic feature define
inter_data_dir = cf.get("acoustic","inter_data_dir")
dimension = cf.getint("acoustic","dimension")
label_style = cf.get("acoustic","label_style")
## feature directory
acoustic_feat_dir = cf.get("acoustic","acoustic_data_dir")
in_stepw_dir = os.path.join(acoustic_feat_dir,"stepw")
in_mgc_dir = os.path.join(acoustic_feat_dir,"mgc")
in_lf0_dir = os.path.join(acoustic_feat_dir,"lf0")
in_bap_dir = os.path.join(acoustic_feat_dir,"bap")
in_sp_dir = os.path.join(acoustic_feat_dir,"sp")
in_seglf0_dir = os.path.join(acoustic_feat_dir,"lf0")
in_acous_feats_dir = os.path.join("./data","in_acoustic_feats")

in_dur_dir = ""
binary_label_dir = os.path.join(inter_data_dir,'binary_label_' + str(dimension))
nn_label_dir = os.path.join(inter_data_dir,'nn_no_silence_lab_')
out_feat_dir = os.path.join(inter_data_dir, 'binary_label_')

nn_label_norm_dir = os.path.join(inter_data_dir,'nn_no_silence_lab_norm_')
label_norm_file = 'label_norm_%s.dat' % (label_style)
label_norm_file = os.path.join(inter_data_dir, label_norm_file)

## global directory
model_dir = os.path.join("./data/path","models")
tf_model_dir =os.path.join("./data/path","models","tensorflow")

var_dir = os.path.join(inter_data_dir, 'var')
if not os.path.exists(var_dir):
    os.makedirs(var_dir)

output_feature_normalisation = "MVN"
_NORM_INFO_FILE_NAME = 'norm_info_%s_%d_%s.dat'
combined_feature_name = ''
for feature_name in output_features:
    combined_feature_name += '_'
    combined_feature_name += feature_name
inp_stats_file = os.path.join(inter_data_dir, "input_%s_%d.norm" %(inp_norm, inp_dim))
out_stats_file = os.path.join(inter_data_dir, "output_%s_%d.norm" %(out_norm, out_dim))

## glottHMM
glott_hmm_path = cf.get("paths","glott_hmm")
glotthmm_in_f0_dir = os.path.join(glott_hmm_path,"f0")
glotthmm_in_gain_dir = os.path.join(glott_hmm_path,"gain")
glotthmm_in_hnr_dir = os.path.join(glott_hmm_path,"hnr")
glotthmm_in_lsf_dir = os.path.join(glott_hmm_path,"lsf")
glotthmm_in_lsfsource_dir = os.path.join(glott_hmm_path,"lsfsource")
# glottDNN
glott_dnn_path = cf.get("paths","glott_dnn")
glottdnn_in_f0_dir = os.path.join(glott_dnn_path,"f0")
glottdnn_in_gain_dir = os.path.join(glott_dnn_path,"gain")
glottdnn_in_hnr_dir = os.path.join(glott_dnn_path,"hnr")
glottdnn_in_lsf_dir = os.path.join(glott_dnn_path,"lsf")
glottdnn_in_slsf_dir= os.path.join(glott_dnn_path,"slsf")
#
in_pdd_dir = os.path.join("./data", 'pdd')

add_frame_features = cf.getboolean("magphase_vocoder","add_frame_features")
subphone_feats = cf.get("labels","subphone_feats")
in_label_align_dir = cf.get("labels","in_label_align_dir")
additional_features = cf.get("labels","additional_features")
silence_pattern = cf.get("labels","silence_pattern")
label_style = cf.get("labels","label_style")
label_type = cf.get("labels","label_type")
#
file_id_scp = cf.get("file","file_id_scp")
test_id_scp = cf.get("file","test_id_scp")
question_file_name = cf.get("file","question_file_name")
#


in_dimension_dict = {}
out_dimension_dict = {}
in_dir_dict = {}
cmp_dim = 0
for feature_name in output_features:
    log.debug(' %s' % feature_name)
    in_dimension = 0
    out_dimension = 0
    in_directory = ''
    if feature_name == 'mgc':
        in_dimension  = vocoder_output_dim.mgc_dim
        out_dimension = vocoder_output_dim.dmgc_dim
        in_directory  = in_mgc_dir
    elif feature_name == 'bap':
        in_dimension = vocoder_output_dim.bap_dim
        out_dimension = vocoder_output_dim.dbap_dim
        in_directory  = in_bap_dir
    elif feature_name == 'lf0':
        in_dimension = vocoder_output_dim.lf0_dim
        out_dimension = vocoder_output_dim.dlf0_dim
        in_directory  = in_lf0_dir
        if waveform.vocoder_type == 'MAGPHASE':
            in_directory = in_acous_feats_dir
    elif feature_name == 'vuv':
        out_dimension = 1
    elif feature_name == 'stepw':
        in_dimension = vocoder_output_dim.stepw_dim
        out_dimension = vocoder_output_dim.stepw_dim
        in_directory  = in_stepw_dir
    elif feature_name == 'sp':
        in_dimension = vocoder_output_dim.sp_dim
        out_dimension = vocoder_output_dim.sp_dim
        in_directory  = in_sp_dir
    elif feature_name == 'seglf0':
        in_dimension = vocoder_output_dim.seglf0_dim
        out_dimension = vocoder_output_dim.seglf0_dim
        in_directory = in_seglf0_dir
    ## for GlottHMM (start)
    elif feature_name == 'F0':
        in_dimension = vocoder_output_dim.F0_dim
        out_dimension = vocoder_output_dim.dF0_dim
        in_directory  = glotthmm_in_f0_dir
    elif feature_name == 'Gain':
        in_dimension = vocoder_output_dim.Gain_dim
        out_dimension = vocoder_output_dim.dGain_dim
        in_directory  = glotthmm_in_gain_dir
    elif feature_name == 'HNR':
        in_dimension = vocoder_output_dim.HNR_dim
        out_dimension = vocoder_output_dim.dHNR_dim
        in_directory  = glotthmm_in_hnr_dir
    elif feature_name == 'LSF':
        in_dimension = vocoder_output_dim.LSF_dim
        out_dimension = vocoder_output_dim.dLSF_dim
        in_directory  = glotthmm_in_lsf_dir
    elif feature_name == 'LSFsource':
        in_dimension = vocoder_output_dim.LSFsource_dim
        out_dimension = vocoder_output_dim.dLSFsource_dim
        in_directory  = glotthmm_in_lsf_dir

    ## for GlottHMM (end)

    ## for GlottDNN (start)
    elif feature_name == 'f0':
        in_dimension = vocoder_output_dim.f0_dim
        out_dimension = vocoder_output_dim.df0_dim
        in_directory  = glottdnn_in_f0_dir
    elif feature_name == 'gain':
        in_dimension = vocoder_output_dim.gain_dim
        out_dimension = vocoder_output_dim.dgain_dim
        in_directory  = glottdnn_in_gain_dir
    elif feature_name == 'hnr':
        in_dimension = vocoder_output_dim.hnr_dim
        out_dimension = vocoder_output_dim.dhnr_dim
        in_directory  = glottdnn_in_hnr_dir
    elif feature_name == 'lsf':
        in_dimension = vocoder_output_dim.lsf_dim
        out_dimension = vocoder_output_dim.dlsf_dim
        in_directory  = glottdnn_in_lsf_dir
    elif feature_name == 'slsf':
        in_dimension = vocoder_output_dim.slsf_dim
        out_dimension = vocoder_output_dim.dslsf_dim
        in_directory  = glottdnn_in_slsf_dir
    ## for GlottDNN (end)

    ## for HMPD (start)
    elif feature_name == 'pdd':
        in_dimension = vocoder_output_dim.pdd_dim
        out_dimension = vocoder_output_dim.dpdd_dim
        in_directory  = in_pdd_dir
    ## for HMPD (end)

    ## For MagPhase Vocoder (start):
    # Note: 'lf0' is set before. See above.
    elif feature_name == 'mag':
        in_dimension  = vocoder_output_dim.mag_dim
        out_dimension = vocoder_output_dim.dmag_dim
        in_directory  = in_acous_feats_dir

    elif feature_name == 'real':
        in_dimension  = vocoder_output_dim.real_dim
        out_dimension = vocoder_output_dim.dreal_dim
        in_directory  = in_acous_feats_dir

    elif feature_name == 'imag':
        in_dimension  = vocoder_output_dim.imag_dim
        out_dimension = vocoder_output_dim.dimag_dim
        in_directory  = in_acous_feats_dir
    ## For MagPhase Vocoder (end)

    ## for joint dur (start)
    elif feature_name == 'dur':
        in_dimension = vocoder_output_dim.dur_dim
        out_dimension = vocoder_output_dim.dur_dim
        in_directory  = in_dur_dir
    else:
        log.critical('%s feature is not supported right now. Please change the configuration.py to support it' %(feature_name))
    log.info('  in_dimension: %d' % in_dimension)
    log.info('  out_dimension : %d' % out_dimension)
    log.info('  in_directory : %s' %  in_directory)
    if in_dimension > 0:
        in_dimension_dict[feature_name] = in_dimension
        if in_directory == '':
            log.critical('please provide the path for %s feature' %(feature_name))
            raise
        if out_dimension < in_dimension:
            log.critical('the dimensionality setting for %s feature is not correct!' %(feature_name))
        in_dir_dict[feature_name] = in_directory

    if out_dimension > 0:
        out_dimension_dict[feature_name] = out_dimension
        cmp_dim += out_dimension

norm_info_file = os.path.join(inter_data_dir,_NORM_INFO_FILE_NAME %(combined_feature_name, vocoder_output_dim.cmp_dim,output_feature_normalisation))
nn_cmp_dir = os.path.join(inter_data_dir,'nn' + combined_feature_name + '_' + str(out_dim))
nn_cmp_norm_dir = os.path.join(inter_data_dir, 'nn_norm' + combined_feature_name + '_' +str(out_dim))
var_file_dict = {}
for feature_name in list(out_dimension_dict.keys()):
    var_file_dict[feature_name] = os.path.join(var_dir,feature_name + '_' + str(out_dimension_dict[feature_name]))