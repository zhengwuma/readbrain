import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import ttest_1samp, ttest_ind
from nilearn.image import index_img
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img, cluster_level_inference
from nilearn import plotting
from nilearn.plotting import plot_glass_brain
from nilearn.image import index_img
from nilearn.image import threshold_img
import matplotlib.pyplot as plt 

DIR = '/Users/tinam/LAMB/conference_paper/sla/results'
os.chdir(DIR)

#################### model_lin stats
#########  L1
corr_bert = np.load('corr_bert.npy')
corr_gpt = np.load('corr_gpt.npy')

In [31]: corr_bert
Out[31]: 
array([ 0.19819568,  0.15253205,  0.20540969,  0.19155661,  0.15263271,
        0.24994219,  0.05093296,  0.22303782,  0.22090575,  0.15698374,
        0.21069215,  0.25935515,  0.21156082,  0.2187631 ,  0.19404954,
        0.22342503,  0.27117282,  0.18293975,  0.16138945,  0.05541047,
        0.06505831,  0.05728159,  0.03389459, -0.03931784])
        
In [32]: corr_gpt
Out[32]: 
array([0.16230162, 0.09619158, 0.22760409, 0.20417943, 0.22282196,
       0.17339782, 0.15607044, 0.11497179, 0.15089188, 0.1488109 ,
       0.16202433, 0.13713137, 0.14291328, 0.15349504, 0.13673324,
       0.13113223, 0.1198461 , 0.11372403, 0.13459997, 0.06546629,
       0.09403123, 0.10850935, 0.07819703, 0.13025439])
       
# one-sample ttest
t_bert = ttest_1samp(corr_bert, 0, axis=0, nan_policy='omit', alternative='two-sided')
In [30]: t_bert
Out[30]: TtestResult(statistic=9.720939740136478, pvalue=6.502340787863103e-10, df=23)

# one-sample ttest
t_gpt = ttest_1samp(corr_gpt, 0, axis=0, nan_policy='omit', alternative='two-sided')
In [38]: t_gpt
Out[38]: TtestResult(statistic=16.958633341139706, pvalue=8.516951703478928e-15, df=23)

# two-sample ttest
t_modelin = ttest_ind(corr_bert, corr_gpt, axis=0, nan_policy='omit', alternative='two-sided')
In [40]: t_modelin
Out[40]: Ttest_indResult(statistic=1.2101103681916543, pvalue=0.11620918252741866)

########################################

DIR = '/Users/tinam/LAMB/conference_paper/sla/results/corr_fmri_lin'
os.chdir(DIR)

n_L1 = 51
n_L2 = 56

corr1 = []
corr2 = []

# L1
for i in range(1, n_L1 + 1):
	if i != 21:
		corr1.append(nib.load('L1_subj%d_corr.nii.gz' % i))

design_matrix1 = pd.DataFrame([1] * len(corr1), columns=['intercept'])
second_level_model = SecondLevelModel()
second_level_model.fit(corr1, design_matrix=design_matrix1)
z_map1 = second_level_model.compute_contrast(output_type='z_score')
stat_img1 = cluster_level_inference(z_map1,threshold=3,alpha=0.05)
nib.save(stat_img1,'stat_img1.nii')

z_map1_thresholded, _ = threshold_stats_img(z_map1, alpha=0.05, height_control="bonferroni")
nib.save(z_map1_thresholded,'/Users/tinam/LAMB/conference_paper/sla/results/zmap1.nii')

display1 = plotting.plot_glass_brain(z_map1_thresholded, colorbar=True, plot_abs=False, display_mode='lzry')
display1.savefig('/Users/tinam/LAMB/conference_paper/sla/results/L1_zmap.png')
display2 = plotting.plot_stat_map(z_map1_thresholded,threshold=_,)
display2.savefig('/Users/tinam/LAMB/conference_paper/sla/results/L1_stat.png')
plt.close()

# L2    
for i in range(1, n_L2 + 1):
  if i != 42:
    corr2.append(nib.load('L2_subj%d_corr.nii.gz' % i))

design_matrix2 = pd.DataFrame([1] * len(corr2), columns=['intercept'])
second_level_model = SecondLevelModel()
second_level_model.fit(corr2, design_matrix=design_matrix2)
z_map2 = second_level_model.compute_contrast(output_type='z_score')
#stat_img2 = cluster_level_inference(z_map2,threshold=3,alpha=0.05)
nib.save(stat_img2,'stat_img2.nii')

z_map2_thresholded, _ = threshold_stats_img(z_map2, alpha=0.05, height_control="bonferroni")
nib.save(z_map2_thresholded,'/Users/tinam/LAMB/conference_paper/sla/results/zmap2.nii')

display2 = plotting.plot_glass_brain(z_map2_thresholded, colorbar=True, plot_abs=False, display_mode='lzry')
display2.savefig('/Users/tinam/LAMB/conference_paper/sla/results/L2_zmap.png')
display2 = plotting.plot_stat_map(z_map2_thresholded,threshold=_,)
display2.savefig('/Users/tinam/LAMB/conference_paper/sla/results/L2_stat.png')
plt.close()

######################################## 

DIR = '/Users/tinam/LAMB/conference_paper/sla'
os.chdir(DIR)

# L1
corr_num1, corr_dur1 = [],[]
n_L1 = 52
n_L2 = 56
for i in range(1,n_L1+1):
	if i not in [52,21]:
		corr = np.load('results/corr_eye_lin/L1_subj%d_corr_eye_lin.npy' %i)
		corr_num1.append(corr[0])
		corr_dur1.append(corr[1])
corr_num1 = np.array(corr_num1)
corr_dur1 = np.array(corr_dur1)
corr_num1 = corr_num1[:, 0]
corr_dur1 = corr_dur1[:, 0]
np.save('results/corr_num1',corr_num1)
np.save('results/corr_dur1',corr_dur1)

In [94]: corr_num1
Out[94]: 
array([0.12295129, 0.10016806, 0.09605621, 0.08911029, 0.09881977,
       0.07867553, 0.10288652, 0.12274305, 0.10617039, 0.14610949,
       0.08580434, 0.08097552, 0.12388857, 0.11091822, 0.16508859,
       0.11064856, 0.110742  , 0.09494161, 0.09863495, 0.05118748,
       0.09144242, 0.08045417, 0.10652535, 0.09327655, 0.12320689,
       0.09740746, 0.06588003, 0.12461992, 0.1110982 , 0.13976127,
       0.07993112, 0.10212794, 0.12267935, 0.07190246, 0.12084176,
       0.09824226, 0.08757212, 0.07222222, 0.08697623, 0.11874531,
       0.07338106, 0.07434643, 0.08724129, 0.04869408, 0.05443501,
       0.08912662, 0.10792128, 0.06121533, 0.08563734, 0.14351049])
       
In [95]: corr_dur1
Out[95]: 
array([0.1253895 , 0.10003028, 0.09491581, 0.08602118, 0.09509101,
       0.08046075, 0.10506424, 0.11857268, 0.10504845, 0.14210888,
       0.08625011, 0.08085786, 0.11710389, 0.1040348 , 0.16462458,
       0.11146067, 0.10483254, 0.09125819, 0.09870865, 0.04937638,
       0.09063162, 0.08418016, 0.10460031, 0.09338257, 0.12063281,
       0.09840904, 0.06544338, 0.12450132, 0.10652523, 0.14569899,
       0.08043418, 0.10810358, 0.1136298 , 0.06956326, 0.11407494,
       0.09594824, 0.08317558, 0.07254539, 0.08585389, 0.11893576,
       0.07363476, 0.07577073, 0.08345261, 0.04904694, 0.05784202,
       0.09391036, 0.10416807, 0.05432863, 0.08443355, 0.14684964])
     
t_num1 = ttest_1samp(corr_num1, 0, axis=0, nan_policy='omit', alternative='two-sided')
In [97]: t_num1
Out[97]: TtestResult(statistic=28.112702661310106, pvalue=3.464649280208529e-32, df=49)

t_dur1 = ttest_1samp(corr_dur1, 0, axis=0, nan_policy='omit', alternative='two-sided')
In [99]: t_dur1
Out[99]: TtestResult(statistic=28.00519558396089, pvalue=4.1353371910067125e-32, df=49)


# L2
corr_num2, corr_dur2 = [],[]
n_L1 = 52
n_L2 = 56
for i in range(1,n_L1+1):
	if i not in [42]:
		corr = np.load('results/corr_eye_lin/L2_subj%d_corr_eye_lin.npy' %i)
		corr_num2.append(corr[0])
		corr_dur2.append(corr[1])
corr_num2 = np.array(corr_num2)
corr_dur2 = np.array(corr_dur2)
corr_num2 = corr_num2[:, 0]
corr_dur2 = corr_dur2[:, 0]
np.save('results/corr_num2',corr_num2)
np.save('results/corr_dur2',corr_dur2)

In [102]: corr_num2
Out[102]: 
array([0.09959187, 0.07781195, 0.08844585, 0.1047924 , 0.17486639,
       0.11606945, 0.12570732, 0.11861868, 0.06519554, 0.12309528,
       0.10734218, 0.13421312, 0.13158152, 0.05415118, 0.10497267,
       0.06983529, 0.12160947, 0.10061398, 0.11962796, 0.1280967 ,
       0.11364654, 0.12562324, 0.11938714, 0.11382571, 0.10690467,
       0.12298649, 0.09858196, 0.12740452, 0.0878504 , 0.11142113,
       0.09750576, 0.09622247, 0.14151463, 0.09887147, 0.12140709,
       0.14365172, 0.11159089, 0.10187029, 0.13361356, 0.12499553,
       0.11588458, 0.09446216, 0.11216576, 0.11242116, 0.09354013,
       0.06903864, 0.07550323, 0.14788593, 0.10942122, 0.13649246,
       0.12144997])

In [103]: corr_dur2
Out[103]: 
array([0.10163632, 0.08108671, 0.08753561, 0.09813755, 0.17509762,
       0.11771766, 0.12483476, 0.11468837, 0.06810582, 0.11977052,
       0.10742994, 0.13772193, 0.12907836, 0.05282738, 0.10548205,
       0.07105995, 0.11972425, 0.10151016, 0.11881698, 0.13584335,
       0.11565103, 0.12958765, 0.12141596, 0.11740337, 0.10454199,
       0.12247216, 0.08993536, 0.13409728, 0.08735838, 0.10783496,
       0.09673998, 0.09708249, 0.13642479, 0.10161252, 0.11596144,
       0.1458736 , 0.11147976, 0.10285982, 0.13284952, 0.12424071,
       0.11883298, 0.0906972 , 0.1096263 , 0.1174807 , 0.09304554,
       0.06973756, 0.07604036, 0.14767685, 0.10698023, 0.1365976 ,
       0.12281224])
       
t_num2 = ttest_1samp(corr_num2, 0, axis=0, nan_policy='omit', alternative='two-sided')
In [105]: t_num2
Out[105]: TtestResult(statistic=35.027769515827664, pvalue=3.780384431893554e-37, df=50)

t_dur2 = ttest_1samp(corr_dur2, 0, axis=0, nan_policy='omit', alternative='two-sided')
In [107]: t_dur2
Out[107]: TtestResult(statistic=34.627609330763214, pvalue=6.566687861824526e-37, df=50)
      
####### two sample t-test     
t_num = ttest_ind(corr_num1, corr_num2, axis=0, nan_policy='omit', alternative='two-sided')
In [116]: t_num
Ttest_indResult(statistic=-2.654800393417061, pvalue=0.009248021404039256)

t_dur = ttest_ind(corr_dur1, corr_dur2, axis=0, nan_policy='omit', alternative='two-sided')
In [38]: t_dur
Out[38]: Ttest_indResult(statistic=-2.8876505784438247, pvalue=0.004766805621006289)

################################################################################ 

corr_bert vs. corr_num1: Ttest_indResult(statistic=5.1250871106022045, pvalue=2.40320316822371e-06)
corr_bert vs. corr_dur1: Ttest_indResult(statistic=5.220552218490032, pvalue=1.656123984068641e-06)
corr_bert vs. corr_num2: Ttest_indResult(statistic=4.223623070024613, pvalue=6.845855164742284e-05)
corr_bert vs. corr_dur2: Ttest_indResult(statistic=4.217240667142466, pvalue=7.003283644950681e-05)

corr_gpt vs. corr_num1: Ttest_indResult(statistic=5.49945354287976, pvalue=5.498648602246337e-07)
corr_gpt vs. corr_dur1: Ttest_indResult(statistic=5.665602977079929, pvalue=2.8236827615990637e-07)
corr_gpt vs. corr_num2: Ttest_indResult(statistic=4.030023136114219, pvalue=0.00013533602576902876)
corr_gpt vs. corr_dur2: Ttest_indResult(statistic=4.012211118609347, pvalue=0.00014396957678920776)













