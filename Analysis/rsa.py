import os
import sys
import pickle
import numpy as np
from scipy.stats import zscore, pearsonr
import skimage.measure
import nibabel as nib

DIR = '/scratch/zhengwuma2/readbrain'
os.chdir(DIR)

subj_id = int(sys.argv[1])
group = sys.argv[2]

# gpt2_attns_base, gpt2_attn_med, gpt2_attn_large, gpt2_attn_xl
attns = pickle.load(open('Analysis/gpt2_attns_base.p','rb')) #snt,layer*head*word_num*word_num 
n_layer = len(attns[0])

centers = np.genfromtxt('Analysis/LMask_centers_lan.csv', dtype=int)
neighbors = np.genfromtxt('Analysis/LMask_neighbors_lan.csv', dtype=int, delimiter=',')

fmri = pickle.load(open('Results/%s_data/%s_subj%d_fmri.p' %(group,group,subj_id),'rb'))
# dim: snt,nword*nword*x*y*z
_,_,x,y,z = fmri[0].shape

corr_attn = []
s = 1
for fmri_snt, attn_snt in zip(fmri,attns):
	print('Processing sentence%d' %s)
	# max pooling on head
	attn_snt = np.squeeze(skimage.measure.block_reduce(attn_snt, (1,16,1,1), np.max))
	n_word = len(fmri_snt)
	fmri_snt = fmri_snt.reshape(n_word,n_word,-1)
	fmri_snt = np.moveaxis(fmri_snt,-1,0)
	fmri_snt = zscore(fmri_snt,axis=None,nan_policy='omit')
	fmri_snt = np.nan_to_num(fmri_snt) #replace nan with 0
	n_voxel = len(fmri_snt)
	corr_attn_snt = np.zeros((n_layer,n_voxel))
	for c, n in zip(centers,neighbors):
		data = fmri_snt[n].mean(axis=0)
		for l in range(n_layer):
			data_lt = np.tril(data)
			attn_snt_lt = np.tril(attn_snt[l,:,:])
			corr_attn_snt[l,c] = pearsonr(data_lt.ravel(),attn_snt_lt.ravel())[0]
	corr_attn.append(corr_attn_snt)
	s += 1

corr_attn = np.mean(corr_attn, axis=0)
corr_attn = np.swapaxes(corr_attn,0,1)
corr_attn = corr_attn.reshape((x,y,z,n_layer))

affine = np.load('Analysis/img_affine.npy')
corr_attn = nib.Nifti1Image(corr_attn,affine=affine)
nib.save(corr_attn, 'Results/corr_fmri_base/%s_subj%d_corr.nii.gz' %(group,subj_id))