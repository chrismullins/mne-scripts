import numpy as np
import os.path as op

import csv

#datadir = '/home/chris/projects/nme-python/Data/TMS_EEG/Practice - June 6'
datadir = '/home/chris/Data/TMS_EEG/Practice - June 6/'
montage_file = 'TMS_EEG_chanlocs.csv'

montage_positions = np.zeros((62, 3))
montage_chnames = []
with open(op.join(datadir, montage_file), 'rt') as csvfile:
    spamreader = csv.DictReader(csvfile, delimiter=',')
    line_num = 0
    for row in spamreader:
        #print(line, row['labels'], row['X'], row['Y'], row['Z'])
        montage_chnames.append(row['labels'])
        montage_positions[line_num,:] = np.array([np.float(row['X']), np.float(row['Y']), np.float(row['Z'])])
        line_num = line_num+1
tms_montage = mne.channels.Montage(pos=montage_positions, ch_names=montage_chnames, kind='TMS Montage', selection=np.arange(1,62))
