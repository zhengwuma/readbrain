import os
import sys
import pickle
import numpy as np
import skimage.measure
from scipy.stats import pearsonr

DIR = '/scratch/zhengwuma2/readbrain'
os.chdir(DIR)

subj_id = int(sys.argv[1])
group = sys.argv[2]

attns = pickle.load(open('Analysis/gpt2_attns_base.p','rb')) #snt,layer*head*nword*nword

sac_num = pickle.load(open('Results/%s_data/%s_subj%d_sac_num.p' %(group,group,subj_id),'rb')) #snt,nword*nword
sac_dur = pickle.load(open('Results/%s_data/%s_subj%d_sac_dur.p' %(group,group,subj_id),'rb')) #snt,nword*nword

corr_num_upper, corr_num_lower, corr_dur_upper, corr_dur_lower  = [],[],[],[]
for attn_snt, sac_num_snt, sac_dur_snt in zip(attns, sac_num, sac_dur):
    # max pooling on head
    attn_snt_max = np.squeeze(skimage.measure.block_reduce(np.tril(attn_snt), (1, 12, 1, 1), np.max))
    #attn_snt_max = np.squeeze(skimage.measure.block_reduce(attn_snt, (1,12,1,1), np.max))
    # gpt2-base: 1,12,1,1 (12 layer); 
	  # gpt2-medium: 1,16,1,1 (24 layer); 
	  # gpt2-large: 1,20,1,1 (36 layer); 
	  # gpt2-xl: 1,25,1,1 (48 layer)
    
	# Get upper and lower triangles of saccade matrices
    sac_num_snt_lower = np.tril(sac_num_snt)
    sac_dur_snt_lower = np.tril(sac_dur_snt)
    corr_num_lower.append([pearsonr(i.flatten(), sac_num_snt_lower.flatten())[0] for i in attn_snt_max])
    corr_dur_lower.append([pearsonr(i.flatten(), sac_dur_snt_lower.flatten())[0] for i in attn_snt_max])

corr_num = np.nanmean(corr_num_lower, axis=0)
corr_dur = np.nanmean(corr_dur_lower, axis=0)
corr = np.vstack([corr_num,corr_dur])    

np.save('Results/corr_eye_base/%s_subj%d_corr_eye' %(group,subj_id), corr)