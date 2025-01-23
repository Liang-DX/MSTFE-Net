import mne
import numpy as np
import scipy.io as io
import os



# TODO Replace your data and label paths.
data_path = ''
all_data_files = [['B0' + str(i) + '0' + str(j) + 'E.gdf' for j in range(4, 6)] for i in range(1, 10)]

label_path = ''
all_label_files = [['B0' + str(i) + '0' + str(j) + 'E.mat' for j in range(4, 6)] for i in range(1, 10)]

save_path = 'dataset/bci_iv_2b/raw'

if not os.path.exists(save_path):
    os.makedirs(save_path)

description_event = {'783': "CueUnknown"}

for sub in range(1, 10):
    print(f'Processing Subject {sub}'.format(sub))
    data_files = all_data_files[sub - 1]
    label_files = all_label_files[sub - 1]
    sub_data = []
    sub_labels = []
    for i in range(len(data_files)):
        raw_data = mne.io.read_raw_gdf(os.path.join(data_path, data_files[i]), preload=True, verbose=False)
        raw_events, all_event_id = mne.events_from_annotations(raw_data)
        raw_data = mne.io.RawArray(raw_data.get_data() * 1e6, raw_data.info)
        raw_data.info['bads'] += ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']
        picks = mne.pick_types(raw_data.info, eeg=True, exclude='bads')
        t_min, t_max = 0, 4
        event_id = dict()
        for event in all_event_id:
            if event in description_event:
                event_id[description_event[event]] = all_event_id[event]
        raw_epochs = mne.Epochs(raw_data, raw_events, event_id, t_min, t_max, proj=True, picks=picks, baseline=None,
                                preload=True)
        data = raw_epochs.get_data()
        data = data[:, :, :-1]
        print(data.shape)
        true_labels = io.loadmat(os.path.join(label_path, label_files[i]))['classlabel']
        print(true_labels.shape)
        sub_data.append(data)
        sub_labels.extend(true_labels)
    sub_data = np.concatenate(sub_data, axis=0)
    sub_labels = np.array(sub_labels)
    datapath = os.path.join(save_path, 'B0' + str(sub) + 'E_data.npy')
    np.save(os.path.join(save_path, 'B0' + str(sub) + 'E_data.npy'), sub_data)
    np.save(os.path.join(save_path, 'B0' + str(sub) + 'E_label.npy'), sub_labels)
    print(f'Data will be saved in{datapath}')