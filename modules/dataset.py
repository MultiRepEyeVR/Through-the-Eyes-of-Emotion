from torch.utils.data import Dataset
import h5py
import os
import torch


class MultiRepresentationDataset(Dataset):
    def __init__(self, file_path, indices, index_info):
        self.file_path = file_path
        self.indices = indices
        self.index_info = index_info

        # Open HDF5 files and store file objects in a dictionary
        self.file_objects = {}
        for subject_name in self.index_info.keys():
            file_path = os.path.join(self.file_path, f'{subject_name}.h5')
            self.file_objects[subject_name] = h5py.File(file_path, 'r')

        # Precompute subject names and adjusted indices for all data indices
        self.data_index_map = {}
        for data_idx in self.indices:
            subject_name, adjusted_idx = self.get_subject_and_adjusted_index(data_idx)
            self.data_index_map[data_idx] = (subject_name, adjusted_idx)

    def get_subject_and_adjusted_index(self, data_idx):
        for subject_name, index_info in self.index_info.items():
            if index_info["min_index"] <= data_idx <= index_info["max_index"]:
                return subject_name, data_idx - index_info["min_index"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        subject_name, adjusted_data_idx = self.data_index_map[data_idx]
        file = self.file_objects[subject_name]

        # Load and normalize eye data
        eye0_data = torch.from_numpy(file['eye0'][adjusted_data_idx])
        eye1_data = torch.from_numpy(file['eye1'][adjusted_data_idx])

        gaze_data = torch.from_numpy(file['gaze_pupil'][adjusted_data_idx])

        # Extract label and convert to a tensor
        label_data = file['labels'][adjusted_data_idx]
        label = torch.tensor(label_data['label'])

        return gaze_data, eye0_data, eye1_data, label