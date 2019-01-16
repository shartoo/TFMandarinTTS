import os
import configparser
from constant import vocoder_output_dim

cf = configparser.ConfigParser()
pwd = os.getcwd()
cf.read(os.path.join(pwd,'./resource/cfgs','cfg.config'))
#
add_frame_features = ""
inp_dim = 425
out_dim = 187
inp_norm = "MINMAX"
out_norm = "MINMAX"
output_features = ['mgc','lf0', 'vuv', 'bap']
gen_wav_features = ['mgc', 'bap', 'lf0']

out_dimension_dict = {}

for feat_name in output_features:
    if out_dimension > 0:
        out_dimension_dict[feature_name] = out_dimension
        cmp_dim += out_dimension

### acoustic feature define
inter_data_dir = cf.get("acoustic","inter_data_dir")
dimension = cf.getint("acoustic","dimension")
lab_dim = cf.getint("acoustic","lab_dim")
label_style = cf.get("acoustic","label_style")
in_dur_dir = ""
binary_label_dir = os.path.join(inter_data_dir,'binary_label_' + str(dimension))
nn_label_dir = os.path.join(inter_data_dir,'nn_no_silence_lab_')
out_feat_dir = os.path.join(inter_data_dir, 'binary_label_')

nn_label_norm_dir = os.path.join(inter_data_dir,'nn_no_silence_lab_norm_')
label_norm_file = 'label_norm_%s_%d.dat' % (label_style, lab_dim)
label_norm_file = os.path.join(inter_data_dir, label_norm_file)

## global directory
inter_data_dir = os.path.join("./data/path","inter_data")
def_inp_dir = os.path.join("./data/path","inp_feat")
def_out_dir = os.path.join("./data/path","out_feat")
model_dir = os.path.join("./data/path","models")
tf_model_dir =os.path.join("./data/path","models","tensorflow")


output_feature_normalisation = "MVN"
_NORM_INFO_FILE_NAME = 'norm_info_%s_%d_%s.dat'
combined_feature_name = ''
for feature_name in output_features:
    combined_feature_name += '_'
    combined_feature_name += feature_name

norm_info_file = os.path.join(inter_data_dir,_NORM_INFO_FILE_NAME %(combined_feature_name, vocoder_output_dim.cmp_dim,output_feature_normalisation))

nn_cmp_dir = os.path.join(inter_data_dir,'nn' + combined_feature_name + '_' + str(vocoder_output_dim.cmp_dim))

stats_dir = os.path.join("./data/path","stats")
inp_stats_file = os.path.join(stats_dir, "input_%s_%d.norm" %(inp_norm, inp_dim))
out_stats_file = os.path.join(stats_dir, "output_%s_%d.norm" %(out_norm, out_dim))

gen_dir = os.path.join("./data/path","gen")
pred_feat_dir = os.path.join("./data/path","pred_feat")
## feature directory
in_stepw_dir = os.path.join("./data/acoustic_features","stepw")
in_mgc_dir = os.path.join("./data/acoustic_features","mgc")
in_lf0_dir = os.path.join("./data/acoustic_features","lf0")
in_bap_dir = os.path.join("./data/acoustic_features","bap")
in_sp_dir = os.path.join("./data/acoustic_features","sp")
in_seglf0_dir = os.path.join("./data/acoustic_features","lf03")

in_acous_feats_dir = os.path.join("./data","in_acoustic_feats")
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

