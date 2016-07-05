import mne
import os.path as op

#datadir = '/home/chris/projects/nme-python/Data/TMS_EEG/Practice - June 6'
datadir = '/home/chris/Data/TMS_EEG/Practice - June 6/'
vhdr = 'TMS_Practice_100A.vhdr'

raw = mne.io.read_raw_brainvision(op.join(datadir, vhdr), preload=True)

#picks = raw.pick_types(raw.info, eeg=True, stim=False)
picks = raw.pick_types(meg=False, eeg=True, eog=False, stim=False)
#events = mne.find_events(raw, stim_channel='STI 014')
events = raw.get_brainvision_events()
event_id = dict(stim_on=1)
tmin = -0.2
tmax = 0.5
baseline = (None, 0) # means from the first instant to t = 0

#epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, preload=False)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, baseline=baseline, preload=False)


print("EPOCHS: {}",epochs)

evoked = epochs['stim_on'].average()
print("EVOKED: {}",evoked)

evoked.plot()
