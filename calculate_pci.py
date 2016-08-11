import mne
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import csv

#datadir = '/home/chris/Data/TMS_EEG/Practice - June 6/'
#datadir = '/home/chris/Data/TMS_EEG/Practice - June 6/'
datadir = '/home/chris/projects/nme-python/Data/TMS_EEG/Practice-June6/'

vhdr = 'TMS_Practice_100A.vhdr'
edf = 'TMS_EEG_preproc.edf'

# Load the raw brainvision first to extract the events
# Figure out how tf to export those with the EDF..
raw_bv = mne.io.read_raw_brainvision(op.join(datadir, vhdr), preload=True)
events_bv = raw_bv.get_brainvision_events()
# Free that memory back up
del raw_bv

# Read the montage file and create a Montage object
montage_file = 'TMS_EEG_chanlocs.csv'
montage_positions = np.zeros((62, 3))
montage_chnames = []
with open(op.join(datadir, montage_file), 'rt') as csvfile:
    spamreader = csv.DictReader(csvfile, delimiter=',')
    line_num = 0
    for row in spamreader:
        montage_chnames.append(row['labels'].strip('\''))
        # Switch Y and X, otherwise you get frontal sensors on right ear and occipital on left ear
        montage_positions[line_num,:] = np.array([np.float(row['Y']), np.float(row['X']), np.float(row['Z'])])
        line_num = line_num+1
tms_montage = mne.channels.Montage(pos=montage_positions, ch_names=montage_chnames, kind='TMS Montage', selection=np.arange(1,62))
dig_montage = digmontage = mne.channels.DigMontage(hsp=montage_positions, point_names=montage_chnames, hpi=np.zeros((0,3)), elp=np.zeros((0,3)))

# Calculate the centroid of all points in montage
x_pos = [p[1] for p in montage_positions]
y_pos = [p[0] for p in montage_positions]
z_pos = [p[2] for p in montage_positions]
montage_centroid = np.array([sum(x_pos) / len(x_pos), sum(y_pos) / len(y_pos), sum(z_pos) / len(z_pos)])

# Now read the edf file which is preprocessed by Ryan Downey
raw_edf = mne.io.read_raw_edf(op.join(datadir, edf), preload=True, stim_channel=62)

# This is probably good to do, although when I use EDF format I need to pass the montage in
# again later or get the 'no digitization points found' error
# Also: figure out the difference between Montage and DigMontage -- we need to use DigMontage
raw_edf.set_montage(dig_montage)


picks = raw_edf.pick_types(meg=False, eeg=True, eog=False, stim=False)
# Use every other event since we only care about trigger "on"
events_trig_on = events_bv[::2]

event_id = dict(stim_on=1)
tmin = -0.2
tmax = 0.5
baseline = (None, 0) # means from the first instant to t = 0

epochs = mne.Epochs(raw_edf, events_trig_on, event_id, tmin, tmax, proj=True, baseline=baseline, preload=False)

evoked = epochs.average()
# Don't normally have to do this for Brainvision
evoked.set_montage(dig_montage)
#evoked.plot_topomap(times=np.linspace(0.00, 0.5, 10), ch_type='eeg')

#sphere_model = mne.bem.make_sphere_model(r0 = 'auto', head_radius = 'auto', info=evoked.info)
# Using 'auto' sets the head radius to 85 meters, center to god-knows-what
# TODO: This should be more principled, for now just use average head size
sphere_model = mne.bem.make_sphere_model(r0 = montage_centroid, head_radius = 177, info=evoked.info)

#times = np.linspace(tmin, tmax, evoked.data.shape[1])
#evoked.plot()
#for ch_name, ch_avg_data in zip(evoked.ch_names, evoked.data):
#    line = plt.plot(times, ch_avg_data, label=ch_name)
#plt.xlabel('Time relative to stimulus (s)')
#plt.ylabel('Avg evoked potential (V)')
#plt.title('Average evoked potential for each EEG lead')
##plt.legend() # legend looks kinda bad
#plt.show()

#fig, anim = evoked.animate_topomap(ch_type='eeg', times=np.linspace(0.00, 0.5, 10), frame_rate=10, show=False)

from mne.datasets import sample
mne_data_path = sample.data_path()
#source_space = mne.setup_source_space(subject='fsaverage', spacing='oct6', subjects_dir=op.join(mne_data_path, 'subjects')) 
# TODO: Should look more like this:
# source_space = mne.read_source_spaces(fname=op.join(mne_data_path, 'subjects/fsaverage/bem/fsaverage-ico-5-src.fif'))
source_space = mne.read_source_spaces(fname='/home/chris/projects/nme-python/mne-venv2/lib/python2.7/site-packages/examples/MNE-sample-data/subjects/fsaverage/bem/fsaverage-oct-6-src.fif')
fwd_model = mne.make_forward_solution(info=evoked.info, src=source_space, bem=sphere_model, eeg=True, meg=False, trans=None)
cov = mne.compute_covariance(epochs, tmax=0)
inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd_model, cov, loose=0.2)
stc = mne.minimum_norm.apply_inverse(evoked=evoked, inverse_operator=inv, lambda2=1./9., method='MNE')
#brain = stc.plot(subjects_dir=op.join(mne_data_path,'subjects'))
#brain.save_movie('stc.mp4')

bootstrap_min_time = -0.2
bootstrap_max_time = -0.1
bootstrap_min_index = int(stc.time_as_index(bootstrap_min_time))
bootstrap_max_index = int(stc.time_as_index(bootstrap_max_time))

def bootstrap_resample(X, n=None):
    """ Bootstrap resample an array_like
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    if n == None:
        n = len(X)
        
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = X[resample_i]
    return X_resample

# 500 Maximum Absolute Values for each dipole
bootstrap_MAVs = np.zeros((stc.data.shape[0], 500))

#for count in range(0,500):
#    # Take the points between -0.2s and 0.0s for each source and generate a bootstrap of them
#    bs_prestimulus = np.apply_along_axis(bootstrap_resample, axis=1, arr=stc.data[:, bootstrap_min_index:bootstrap_max_index])
#    bootstrap_MAVs[:, count] = np.sum(bs.prestimulus, axis=1)
for count in range(0,500):
    # Take the points between -0.2s and 0.0s for each source and generate a bootstrap of them
    bs_prestimulus = np.apply_along_axis(bootstrap_resample, axis=1, arr=stc.data[:, bootstrap_min_index:bootstrap_max_index])
    bootstrap_MAVs[:, count] = np.max(np.abs((bs_prestimulus)), axis=1)
    print("Done with {}".format(count))

bootstrap_99_percentiles = np.zeros((stc.data.shape[0], 1))
bootstrap_99_percentiles[:, 0] = np.apply_along_axis(np.percentile, axis=1, arr=bootstrap_MAVs, q=50)
post_stim_min_index = int(stc.time_as_index(0.0))
post_stim_max_index = int(stc.time_as_index(0.3))
significant_activations = stc.data[:, post_stim_min_index:post_stim_max_index] > bootstrap_99_percentiles
significant_activations_sorted = significant_activations.copy()
#significant_activations_sorted.sort(axis=0)

import zlib 
def kolmogorov(s):
    l = float(len(s))
    compr = zlib.compress(s)
    c = float(len(compr))
    return c/l 

#sig_act_sorted_string = ''.join(significant_activations_sorted.astype(int).astype(str).ravel().tolist())
sig_act_sorted_str = significant_activations_sorted.astype(int).ravel().tostring()
print("Final PCI: {}".format(kolmogorov(sig_act_sorted_str)))
