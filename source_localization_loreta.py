import mne
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import csv

from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              write_inverse_operator)

#datadir = '/home/chris/projects/nme-python/Data/TMS_EEG/Practice - June 6'
datadir = '/home/chris/Data/TMS_EEG/Practice - June 6/'
vhdr = 'TMS_Practice_100A.vhdr'

raw = mne.io.read_raw_brainvision(op.join(datadir, vhdr), preload=True)

montage_file = 'TMS_EEG_chanlocs.csv'

montage_positions = np.zeros((62, 3))
montage_chnames = []
with open(op.join(datadir, montage_file), 'rt') as csvfile:
    spamreader = csv.DictReader(csvfile, delimiter=',')
    line_num = 0
    for row in spamreader:
        montage_chnames.append(row['labels'].strip('\''))
        montage_positions[line_num,:] = np.array([np.float(row['X']), np.float(row['Y']), np.float(row['Z'])])
        line_num = line_num+1
tms_montage = mne.channels.Montage(pos=montage_positions, ch_names=montage_chnames, kind='TMS Montage', selection=np.arange(1,62))

raw.set_montage(tms_montage)

picks = raw.pick_types(meg=False, eeg=True, eog=False, stim=False)
events = raw.get_brainvision_events()
# Use every other event since we only care about trigger "on"
events_trig_on = events[::2]
event_id = dict(stim_on=1)
tmin = -0.2
tmax = 0.5
baseline = (None, 0) # means from the first instant to t = 0

epochs = mne.Epochs(raw, events_trig_on, event_id, tmin, tmax, proj=True, baseline=baseline, preload=False)

evoked = epochs.average()
#evoked.plot_topomap(times=np.linspace(0.00, 0.5, 10), ch_type='eeg')

noise_cov = mne.compute_covariance(epochs, tmax=0., method=['shrunk', 'empirical'])
fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, raw.info)


