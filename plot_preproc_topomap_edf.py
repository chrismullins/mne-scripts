import mne
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import csv

datadir = '/home/chris/Data/TMS_EEG/Practice - June 6/'
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
        montage_positions[line_num,:] = np.array([np.float(row['X']), np.float(row['Y']), np.float(row['Z'])])
        line_num = line_num+1
tms_montage = mne.channels.Montage(pos=montage_positions, ch_names=montage_chnames, kind='TMS Montage', selection=np.arange(1,62))

# Now read the edf file which is preprocessed by Ryan Downey
raw_edf = mne.io.read_raw_edf(op.join(datadir, edf), preload=True, stim_channel=62)

# This is probably good to do, although when I use EDF format I need to pass the montage in
# again later or get the 'no digitization points found' error
raw_edf.set_montage(tms_montage)


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
evoked.set_montage(tms_montage)
#evoked.plot_topomap(times=np.linspace(0.00, 0.5, 10), ch_type='eeg')

#times = np.linspace(tmin, tmax, evoked.data.shape[1])
#evoked.plot()
#for ch_name, ch_avg_data in zip(evoked.ch_names, evoked.data):
#    line = plt.plot(times, ch_avg_data, label=ch_name)
#plt.xlabel('Time relative to stimulus (s)')
#plt.ylabel('Avg evoked potential (V)')
#plt.title('Average evoked potential for each EEG lead')
##plt.legend() # legend looks kinda bad
#plt.show()

#evoked.animate_topomap(ch_type='eeg', times=np.arange(0.05,0.15,0.01), frame_rate=10)

