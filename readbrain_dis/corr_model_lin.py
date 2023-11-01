import os
import sys
import pickle
import numpy as np
import skimage.measure
from scipy.stats import pearsonr

DIR = '/scratch/zhengwuma2/sla'
os.chdir(DIR)

attns_gpt = pickle.load(open('analysis/gpt2_attns_med.p','rb')) #snt,layer*head*nword*nword
attns_bert = pickle.load(open('analysis/bert_attns.p','rb'))
dis = pickle.load(open('analysis/dis.pkl', 'rb'))  # snt, nword*nword

# Initialize corr_dis for each model
corr_dis_gpt = []
corr_dis_bert = []

# gpt2-med
for attn_snt, dis_snt in zip(attns_gpt, dis):
    attn_snt_max = np.squeeze(skimage.measure.block_reduce(np.tril(attn_snt), (1, 24, 1, 1), np.max))    
    dis_snt_lower = np.tril(dis_snt)
    corr_dis_gpt.append([pearsonr(i.flatten(), dis_snt_lower.flatten())[0] for i in attn_snt_max])
corr_dis_gpt = np.nanmean(corr_dis_gpt, axis=0)
np.save('results/corr_model_gpt/corr_gpt', corr_dis_gpt)
print('finish1')

# bert-large
for attn_snt, dis_snt in zip(attns_bert, dis):
    attn_snt_max = np.squeeze(skimage.measure.block_reduce(np.tril(attn_snt), (1, 24, 1, 1), np.max))    
    dis_snt_lower = np.tril(dis_snt)
    corr_dis_bert.append([pearsonr(i.flatten(), dis_snt_lower.flatten())[0] for i in attn_snt_max])
corr_dis_bert = np.nanmean(corr_dis_bert, axis=0) 
np.save('results/corr_model_bert/corr_bert', corr_dis_bert)
print('finish2')
