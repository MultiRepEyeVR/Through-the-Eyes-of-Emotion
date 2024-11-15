import json
import os
from sklearn.model_selection import train_test_split


def load_index_info(data_path):
    with open(data_path, 'r') as file:
        index_info = json.load(file)
    return index_info


def create_index_mapping(subjects, index_info):
    indices = []
    for subject in subjects:
        if subject in index_info:
            indices.extend(range(index_info[subject]["min_index"], index_info[subject]["max_index"] + 1))
    return indices


def create_train_labels(indices_list, subjects, index_info, with_surprise):
    index_label_map = {}

    for subject in subjects:
        if subject in index_info:
            subject_info = index_info[subject]
            current_index = subject_info["min_index"]

            for i in range(7):  # known that there are 7 sessions (0-6)
                if not with_surprise and i == 1:
                    continue
                session_keys = [f"{i}a_amount", f"{i}b_amount"]
                total_amount = sum(subject_info[key] for key in session_keys)

                for idx in range(current_index, current_index + total_amount):
                    index_label_map[idx] = i

                current_index += total_amount

    label_list = [index_label_map.get(idx) for idx in indices_list if idx in index_label_map]

    return label_list


def get_test_val_train_indices(test_subjects, train_subjects, data_path, train_percentage=0.1, with_surprise=True):
    index_info = load_index_info(os.path.join(data_path, 'index_info.json'))

    test_indices = create_index_mapping(test_subjects, index_info)

    full_train_indices = create_index_mapping(train_subjects, index_info)

    full_train_labels = create_train_labels(full_train_indices, train_subjects, index_info, with_surprise)

    train_indices, val_indices = train_test_split(full_train_indices, test_size=1.0 - train_percentage, stratify=full_train_labels, random_state=0)

    return test_indices, val_indices, train_indices, index_info