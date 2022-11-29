import h5py
import math
import pandas as pd
from tensorflow.keras.utils import Sequence
import numpy as np


class ECGSequence(Sequence):
    @classmethod
    def get_train_and_val(cls, hdf5_dset, path_to_csv, batch_size=8,  val_split=0.02):


        # Loop for n samples
        f_i = []
        s = 0
        for i in range(-1, 5):
            f_n = h5py.File('data_train/exams_part' + str(i+1) + '.hdf5', 'r')
            f_i.append(f_n)
            x_n = f_n[hdf5_dset]
            # x_i.append(len(x_n))
            s = s + len(x_n)
        n_samples = s - len(f_i)

        n_train = math.ceil(n_samples*(1-val_split))
        print(n_samples-n_train)
        train_seq = cls(hdf5_dset, path_to_csv, batch_size, end_idx=n_train)
        valid_seq = cls(hdf5_dset, path_to_csv, batch_size, start_idx=n_train)
        print(len(valid_seq))  
        return train_seq, valid_seq

    def __init__(self, hdf5_dset, path_to_csv=None, batch_size=8,
                 start_idx=0, end_idx=None):
        if path_to_csv is None:
            self.y = None
        else:
        
            df = pd.read_csv(path_to_csv)
            ls_y = []
            for i in range(-1, 5):
                df_y = df[df['trace_file'] == 'exams_part' + str(i+1) + '.hdf5']
                df_y = df_y.sort_values(by = ['exam_id'])
                df_y = df_y[["1dAVb","RBBB","LBBB","SB","ST","AF"]]
                ls_y.append(df_y)
            df = pd.concat(ls_y)
            self.y = df.values

        # Loop for tracings

        x_tracings = []
        for i in range(-1, 5):
            f = h5py.File('data_train/exams_part' + str(i+1) + '.hdf5', 'r')
            exam_id = f['exam_id']
            exam_id = np.asarray(exam_id)
            x0 = f[hdf5_dset]
            x0 = np.asarray(x0)
            x0 = x0[np.argsort(exam_id)]
            x0 = x0[1:]
            x_tracings.append(x0)
        self.x = np.concatenate(tuple(x_tracings))




        self.batch_size = batch_size
        # print(self.batch_size)
        if end_idx is None:
            end_idx = len(self.x)
        self.start_idx = start_idx
        self.end_idx = end_idx
        # print(self.start_idx)
        # print(self.end_idx)

    @property
    def n_classes(self):
        return self.y.shape[1]

    def __getitem__(self, idx):
        start = self.start_idx + idx * self.batch_size
        end = min(start + self.batch_size, self.end_idx)
        # a = np.asarray(self.y[start:end]).dtype #Kim
        # print(self.y)
        if self.y is None:
            return np.array(self.x[start:end, :, :])
        else:
            return np.array(self.x[start:end, :, :]), np.array(self.y[start:end])

    def __len__(self):
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)

    def __del__(self):
        self.f.close()

class ECGSequence_test(Sequence):
    @classmethod
    def get_train_and_val(cls, hdf5_dset, path_to_csv=None, batch_size=8,  val_split=0.02):
       
        f_i = []
        s = 0
        for i in range(0, 1):
            f_n = h5py.File('data/ecg_tracings_' + str(i+1) + '.hdf5', 'r')
            f_i.append(f_n)
            x_n = f_n[hdf5_dset]
            # x_i.append(len(x_n))
            s = s + len(x_n)
        n_samples = s - len(f_i)
        print('len n test samples:', n_samples)

        n_train = math.ceil(n_samples*(1-val_split))

        # print(n_samples-n_train)
        train_seq = cls(hdf5_dset, path_to_csv, batch_size, end_idx=n_train)
        valid_seq = cls(hdf5_dset, path_to_csv, batch_size, start_idx=n_train)
        # print('len')
        # print(len(valid_seq))  
        return train_seq, valid_seq

    def __init__(self, hdf5_dset, path_to_csv=None, batch_size=8,
                 start_idx=0, end_idx=None):
        if path_to_csv is None:
            self.y = None    
        else:
            self.y = pd.read_csv(path_to_csv).values # Paper

        self.f = h5py.File('data/ecg_tracings_1.hdf5', 'r')
        self.x = self.f[hdf5_dset]

        self.batch_size = batch_size
        # print(self.batch_size)
        if end_idx is None:
            end_idx = len(self.x)
        self.start_idx = start_idx
        self.end_idx = end_idx
    @property
    def n_classes(self):
        return self.y.shape[1]

    def __getitem__(self, idx):
        start = self.start_idx + idx * self.batch_size
        end = min(start + self.batch_size, self.end_idx)
        # a = np.asarray(self.y[start:end]).dtype #Kim
        # print(self.y)
        if self.y is None:
            return np.array(self.x[start:end, :, :])
        else:
            return np.array(self.x[start:end, :, :]), np.array(self.y[start:end])

    def __len__(self):
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)

    def __del__(self):
        self.f.close()
