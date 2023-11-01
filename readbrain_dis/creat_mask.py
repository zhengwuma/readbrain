import nibabel as nib
from nilearn import datasets, image, plotting
from nilearn.image import resample_img
from rsatoolbox.util.searchlight import get_volume_searchlight

# creat language mask
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm', symmetric_split=True)
LIFG = [i for i in atlas.labels if 'Left Inferior Frontal Gyrus' in i]
LAG = ['Left Angular Gyrus']
LTemp = [i for i in atlas.labels if 'Left' in i and 'Temporal Gyrus' in i]
LMask = LTemp + LIFG + LAG 
roi_val = [atlas.labels.index(roi) for roi in LMask]
mask = image.math_img('np.isin(img, {}).astype(int)'.format(roi_val), img=atlas.maps)

# plot to check
plotting.plot_glass_brain(mask, title='LMask')
plotting.show()

# save mask
mask.to_filename('LMask.nii.gz')

# load mask
mask = nib.load('LMask.nii.gz')

# subset fMRI data using mask
data_file = 'sub-01/func/sub-01_task-read_run-3_bold.nii.gz'
img = image.smooth_img(data_file,[2,2,2]) #fwhm smoothed
mask = resample_img(mask,target_affine=img.affine,target_shape=img.shape[:-1])
centers, neighbors = get_volume_searchlight(mask.get_fdata(), radius=5, threshold=0.5)

# save centers and neighbors
np.savetxt('lan_centers.csv', centers, fmt='%d')
np.savetxt('lan_neighbors.csv', neighbors, fmt='%d', delimiter=',')