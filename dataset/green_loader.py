import random
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Dataset
from config.config import Config


class TrajDataset(Dataset):
    def __init__(self, traj_data):
        self.traj_data = traj_data

    def __len__(self):
        return len(self.traj_data)

    def __getitem__(self, idx):
        return self.traj_data.iloc[idx]

def continuous_mask_sequence(sequence, mask_percentage, mask_length):
    num_elements = len(sequence)
    num_elements_to_mask = int(num_elements * mask_percentage)
    masked_sequence = sequence.clone()

    num_continuous_intervals = num_elements - mask_length + 1

    num_intervals_to_mask = num_elements_to_mask // mask_length

    max_interval = num_continuous_intervals - num_intervals_to_mask

    start_indices = set()
    # TODO
    # num = 0
    while len(start_indices) < num_intervals_to_mask:
        start_idx = random.randint(0, max_interval)
        if all(start_idx < s or start_idx > s + mask_length for s in start_indices):
            start_indices.add(start_idx)
        # num += 1

    for start_idx in start_indices:
        masked_sequence[start_idx:start_idx+mask_length] = Config.road_special_tokens['mask_token']

    return masked_sequence


class TrajDataLoader:
    def __init__(self):
        self.batch_size = Config.batch_size
        self.num_workers = 8

    def get_data_loader(self, traj_data, is_shuffle=False):
        dataset = TrajDataset(traj_data=traj_data)

        data_loader = DataLoader(dataset,
                                 batch_size=self.batch_size,
                                 shuffle=is_shuffle,
                                 num_workers=self.num_workers,
                                 collate_fn=self._collate_func)
        return data_loader

    def _collate_func(self, data_df):
        bz = len(data_df)
        road_traj_list = [traj.road_traj for traj in data_df]
        grid_traj_list = [traj.grid_traj for traj in data_df]

        road_temporal_list = [traj.ptime for traj in data_df]

        grid_feature_list = [traj.grid_feature for traj in data_df]

        road_lens = [len(path) for path in road_traj_list]
        grid_lens = [len(grid) for grid in grid_traj_list]
        max_road_len = max(road_lens)
        max_grid_len = max(grid_lens)

        road_traj_inputs = torch.zeros((bz, max_road_len + 1), dtype=torch.long)
        grid_traj_inputs = torch.zeros((bz, max_grid_len + 1), dtype=torch.long)
        mask_road_traj_inputs = torch.zeros((bz, max_road_len + 1), dtype=torch.long)
        road_type_inputs = torch.zeros((bz, max_road_len + 1), dtype=torch.long)

        grid_feature_inputs = torch.zeros((bz, max_grid_len + 1, 4), dtype=torch.float32)

        grid_time_emb_inputs = torch.zeros((bz, max_grid_len + 1, Config.hidden_emb_dim), dtype=torch.float32)

        road_week_inputs = torch.zeros((bz, max_road_len + 1), dtype=torch.long)
        road_minute_inputs = torch.zeros((bz, max_road_len + 1), dtype=torch.long)


        for i in range(bz):
            path = road_traj_list[i]
            road_len = len(path)
            grid = grid_traj_list[i]
            grid_len = len(grid)

            grid_temporal_emb = torch.tensor(data_df[i]['time_emb'], dtype=torch.float32)

            road_type = data_df[i]['road_type']
            road_type_inputs[i, 1:road_len + 1] = torch.LongTensor(road_type) + 1

            grid_feature_inputs[i, 1:grid_len + 1] = torch.tensor(grid_feature_list[i], dtype=torch.float32)

            road_shift_with_tokens = torch.LongTensor(path) + len(Config.road_special_tokens)
            road_traj_inputs[i, 1:road_len + 1] = road_shift_with_tokens
            road_traj_inputs[i, 0] = Config.road_special_tokens['cls_token']

            mask_road_traj = continuous_mask_sequence(road_shift_with_tokens,
                                                      Config.mask_ratio,
                                                      Config.mask_length)

            mask_road_traj_inputs[i, 1:road_len + 1] = mask_road_traj
            mask_road_traj_inputs[i, 0] = Config.road_special_tokens['cls_token']

            grid_shift_with_tokens = torch.LongTensor(grid) + len(Config.grid_special_tokens)
            grid_traj_inputs[i, 1:grid_len + 1] = grid_shift_with_tokens
            grid_traj_inputs[i, 0] = Config.grid_special_tokens['cls_token']

            grid_time_emb_inputs[i, 0] = grid_temporal_emb[0]
            grid_time_emb_inputs[i, 1:grid_len + 1] = grid_temporal_emb

            road_temporal = road_temporal_list[i]
            road_date = [datetime.fromtimestamp(t) for t in road_temporal]
            road_weeks = [d.weekday() + 1 for d in road_date]
            road_minutes = [d.minute + 1 + d.hour * 60 for d in road_date]

            road_week_inputs[i, 1:road_len + 1] = torch.LongTensor(road_weeks)
            road_week_inputs[i, 0] = road_weeks[0]
            road_minute_inputs[i, 1:road_len + 1] = torch.LongTensor(road_minutes)
            road_minute_inputs[i, 0] = road_minutes[0]

        mask_road_index = torch.where(mask_road_traj_inputs == Config.road_special_tokens['mask_token'])

        road_data = {
            'road_traj': road_traj_inputs,
            'mask_road_index': mask_road_index,
            'road_type': road_type_inputs,
            'road_weeks': road_week_inputs,
            'road_minutes': road_minute_inputs,
        }

        grid_data = {
            'grid_traj': grid_traj_inputs,
            'grid_feature': grid_feature_inputs,
            'grid_time_emb': grid_time_emb_inputs,
        }
        return road_data, grid_data
