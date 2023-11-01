import os
import sys
import pickle
import numpy as np
from scipy.stats import pearsonr

DIR = '/scratch/zhengwuma2/sla'
os.chdir(DIR)

subj_id = int(sys.argv[1])
group = sys.argv[2]

dis = pickle.load(open('analysis/dis.pkl','rb')) #snt,nword*nword
sac_num = pickle.load(open('/scratch/zhengwuma2/readbrain/Results/%s_data/%s_subj%d_sac_num.p' %(group,group,subj_id),'rb')) #snt,nword*nword
sac_dur = pickle.load(open('/scratch/zhengwuma2/readbrain/Results/%s_data/%s_subj%d_sac_dur.p' %(group,group,subj_id),'rb')) #snt,nword*nword

corr_num_lower, corr_dur_lower = [],[]

for dis_snt, sac_num_snt, sac_dur_snt in zip(dis, sac_num, sac_dur): 
    dis_snt_lower = np.tril(dis_snt)
    sac_num_snt_lower = np.tril(sac_num_snt)
    sac_dur_snt_lower = np.tril(sac_dur_snt)
    corr_num_lower.append(pearsonr(dis_snt_lower.flatten(), sac_num_snt_lower.flatten())[0])
    corr_dur_lower.append(pearsonr(dis_snt_lower.flatten(), sac_dur_snt_lower.flatten())[0])

corr_num = np.nanmean(corr_num_lower, axis=0)
corr_dur = np.nanmean(corr_dur_lower, axis=0)
corr = np.vstack([corr_num,corr_dur])    
np.save('results/corr_eye_lin/%s_subj%d_corr_eye_lin' %(group,subj_id), corr)
