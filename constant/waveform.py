vocoder_type = "STRAIGHT"
samplerate = 48000
framelength = 4096
frameshift = 1000 * 240 / 48000
sp_dim = (4096 / 2) + 1
fw_alpha = 0.77
postfilter_coef = 1.4
minimum_phase_order = 2047
use_cep_ap = True
do_post_filtering =True
apply_GV = False