import os
import configparser

cf = configparser.ConfigParser()
pwd = os.getcwd()
cf.read(os.path.join(pwd,'./resource/cfgs','cfg.config'))

### acoustic feature define
inter_data_dir = cf.get("acoustic","inter_data_dir")
dimension = cf.getint("acoustic","dimension")
lab_dim = cf.getint("acoustic","lab_dim")
label_style = cf.get("acoustic","label_style")

binary_label_dir = os.path.join(inter_data_dir,'binary_label_' + str(dimension))
nn_label_dir = os.path.join(inter_data_dir,'nn_no_silence_lab_' + suffix)
nn_label_norm_dir = os.path.join(inter_data_dir,'nn_no_silence_lab_norm_' + suffix)
label_norm_file = 'label_norm_%s_%d.dat' % (label_style, lab_dim)
label_norm_file = os.path.join(inter_data_dir, label_norm_file)

output_features = ['mgc','lf0', 'vuv', 'bap']
gen_wav_features = ['mgc', 'bap', 'lf0']

## global directory
inter_data_dir = os.path.join("./data/path","inter_data")
def_inp_dir = os.path.join("./data/path","inp_feat")
def_out_dir = os.path.join("./data/path","out_feat")
model_dir = os.path.join("./data/path","models")
stats_dir = os.path.join("./data/path","stats")
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


#
file_id_scp = os.path.join("./data", 'file_id_list.scp')
test_id_scp = os.path.join("./data", 'test_id_scp.scp')
#
inp_dim = 425
out_dim = 187
inp_norm = "MINMAX"
out_norm = "MINMAX"
