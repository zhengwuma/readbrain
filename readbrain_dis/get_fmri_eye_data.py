import sys
import os
import pickle
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn import image

DIR = '/scratch/zhengwuma2/readbrain'
os.chdir(DIR)

subj_id = int(sys.argv[1])
group = sys.argv[2]

subj = 'sub-0%d' %subj_id if subj_id < 10 else 'sub-%d' %subj_id

# words = pd.read_csv('Analysis/words.csv')
# words['Word'] = snts['Word'].str.replace(r'[^\w\s]+', '')
# words['Word'] = snts['Word'].str.lower()
# words.to_csv('words.csv', index=False)

words = pd.read_csv('Analysis/words.csv')

# # check number of words
# n_word = 0
# for s in snts:
# 	n_word += len(s.split())
# assert len(words) == n_word 

fmri, sac_num, sac_dur = [],[],[]
for article in range(1,6):
	run = 1
	events = pd.read_csv('Data/%s/%s/func/%s_task-read_run-%d_events.tsv' %(group,subj,subj,run), delimiter='\t')
	# remove NAN in event
	events = events.dropna().reset_index(drop=True)
	run_article = int(events.SentenceID[0].split('.')[1])
	while run_article != article and run < 5:
		run += 1
		events = pd.read_csv('Data/%s/%s/func/%s_task-read_run-%d_events.tsv' %(group,subj,subj,run), delimiter='\t')
		events = events.dropna().reset_index(drop=True)
		run_article = int(events.SentenceID[0].split('.')[1])
		
	n_snts = int(words[words.SentenceID.str.match('t.0%d' %article)].SentenceID.iloc[-1].split('.')[-1])

	# subset fMRI data using mask
	fmri_file = 'Data/%s/%s/func/%s_task-read_run-%d_bold.nii.gz' %(group,subj,subj,run)
	img = image.smooth_img(fmri_file,[2,2,2]) #fwhm smoothed
	img_data = img.get_fdata()
	x,y,z,_ = img_data.shape

	for i in range(1,n_snts+1):
		print('Processing article %d, sentence %d' %(article,i))
		snt_id = 't.0%d.0%d' %(article,i) if i < 10 else 't.0%d.%d' %(article,i)
		snt_len = len(words[words.SentenceID.str.match(snt_id)])
		event = events[events.SentenceID.str.match(snt_id)]
		
		# remove self-to-self saccade and add duration
		event['duplicate'] = event.CURRENT_FIX_INTEREST_AREA_ID.eq(event.CURRENT_FIX_INTEREST_AREA_ID.shift())
		for ind, e in event.iterrows():
			if e['duplicate']:
				event.at[ind-1,'duration'] += e.duration
		event = event.drop(event[event['duplicate']==True].index).reset_index(drop=True)

		# get fmri and saccade data		
		fmri_s = np.zeros((snt_len,snt_len,x,y,z))
		sac_num_s = np.zeros((snt_len,snt_len))
		sac_dur_s = np.zeros((snt_len,snt_len))
		for ind, e in event.iterrows():
			if ind < len(event)-1:
				row = int(e.CURRENT_FIX_INTEREST_AREA_ID)-1
				col = int(event.iloc[ind+1].CURRENT_FIX_INTEREST_AREA_ID)-1
				scan = int(np.ceil((event.iloc[ind+1].onset/1000+5)/0.4))
				while scan >= img_data.shape[3]:
					scan-=1
				fmri_s[row,col] = img_data[:,:,:,scan]
				sac_num_s[row,col] += 1
				sac_dur_s[row,col] += e.duration + event.iloc[ind+1].duration
		fmri.append(fmri_s)
		sac_num.append(sac_num_s)
		sac_dur.append(sac_dur_s)

pickle.dump(fmri, open('Results/%s_subj%d_fmri.p' %(group,subj_id),'wb'), protocol=2)
pickle.dump(sac_num, open('Results/%s_subj%d_sac_num.p' %(group,subj_id),'wb'), protocol=2)
pickle.dump(sac_dur, open('Results/%s_subj%d_sac_dur.p' %(group,subj_id),'wb'), protocol=2)



	
	
		
		
	
	
	
	
	
