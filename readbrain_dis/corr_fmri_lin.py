import os
import sys
import pickle
import numpy as np
from scipy.stats import zscore, pearsonr
import skimage.measure
import nibabel as nib

DIR = '/scratch/zhengwuma2/sla'
os.chdir(DIR)

subj_id = int(sys.argv[1])
group = sys.argv[2]

centers = np.genfromtxt('analysis/lan_centers.csv', dtype=int)
neighbors = np.genfromtxt('analysis/lan_neighbors.csv', dtype=int, delimiter=',')

dis = pickle.load(open('analysis/dis.pkl', 'rb'))  # snt, nword*nword
fmri = pickle.load(open('/scratch/zhengwuma2/readbrain/Results/%s_data/%s_subj%d_fmri.p' % (group, group, subj_id), 'rb'))

# dim: snt, nword*nword*x*y*z
_, _, x, y, z = fmri[0].shape

corr_lin = []
s = 1
for fmri_snt, dis_snt in zip(fmri, dis):
    print('Processing sentence%d' % s)
    # max pooling on head
    n_word = len(fmri_snt)
    fmri_snt = fmri_snt.reshape(n_word, n_word, -1)
    fmri_snt = np.moveaxis(fmri_snt, -1, 0)
    fmri_snt = zscore(fmri_snt, axis=None, nan_policy='omit')
    fmri_snt = np.nan_to_num(fmri_snt)  # replace nan with 0
    n_voxel = len(fmri_snt)
    corr_lin_snt = np.zeros((n_voxel,))
    for c, n in zip(centers, neighbors):
        data = fmri_snt[n].mean(axis=0)
        data_lt = np.tril(data)
        dis_snt_lt = np.tril(dis_snt[:, :])
        corr_lin_snt[c] = pearsonr(data_lt.ravel(), dis_snt_lt.ravel())[0]
    corr_lin.append(corr_lin_snt)
    s += 1
print('finish1')

corr_lin = np.mean(corr_lin, axis=0)
corr_lin = corr_lin.reshape((x, y, z))
print('finish2')

affine = np.load('analysis/img_affine.npy')
corr_lin = nib.Nifti1Image(corr_lin, affine=affine)
nib.save(corr_lin, 'results/corr_fmri_lin/%s_subj%d_corr.nii.gz' % (group, subj_id))
print('finish3')
