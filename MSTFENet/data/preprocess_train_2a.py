import mne
import numpy as np
import os

# TODO Replace your data and label paths.

data_path = 'D:\DeepLearning\lightconv1\dataset'
data_files = ['A0'+str(i)+'T.gdf' for i in range(1,10)]

save_path = 'D:\DeepLearning\EEG-TransNet\dataset\\bci_iv_2a'

description_event = {'769':"CueLeft", '770':"CueRight", '771':"CueFoot", '772':"CueTongue"}

for file in data_files:
    raw_data = mne.io.read_raw_gdf(os.path.join(data_path, file), preload=True, verbose=False)
    raw_events, all_event_id = mne.events_from_annotations(raw_data)
    raw_data = mne.io.RawArray(raw_data.get_data()*1e6, raw_data.info)
    raw_data.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']
    picks = mne.pick_types(raw_data.info, eeg=True, exclude='bads')
    t_min, t_max = 0, 4
    event_id = dict()

    for event in all_event_id:
        if event in description_event:
            event_id[description_event[event]] = all_event_id[event]

    raw_epochs = mne.Epochs(raw_data, raw_events, event_id, t_min, t_max, proj=True, picks=picks, baseline=None, preload=True)
    true_labels = raw_epochs.events[:, -1] - event_id['CueLeft'] + 1
    data = raw_epochs.get_data() 
    data = data[:, :, :-1]
    np.save(os.path.join(save_path, file[:-4]+'_data.npy'), data)
    np.save(os.path.join(save_path, file[:-4]+'_label.npy'), true_labels)

