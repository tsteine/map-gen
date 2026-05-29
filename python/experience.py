import torch
import dataclasses

import os
import safetensors.torch
from env import Actions


class ExperienceStorage:
    def __init__(self, num_rooms, data_path, episodes_per_file):
        self.num_rooms = num_rooms
        self.data_path = data_path
        self.episodes_per_file = episodes_per_file
        self.num_files = 0
        os.makedirs(data_path, exist_ok=True)

    def store(self, actions: Actions):
        next_file_number = self.num_files
        assert actions.room_idx.shape[0] == self.episodes_per_file
        file_path = os.path.join(self.data_path, "{}.safetensors".format(next_file_number))
        safetensors.torch.save_file(dataclasses.asdict(actions), file_path)
        self.num_files += 1

    def read_files(self, file_num_list, episodes_per_file):
        data_list = []
        for file_num in file_num_list:
            file_path = os.path.join(self.data_path, "{}.safetensors".format(file_num))
            data = Actions(**safetensors.torch.load_file(file_path))
            ind = torch.randperm(data.room_idx.shape[0])[:episodes_per_file]
            data = Actions(
                room_idx=data.room_idx[ind],
                room_x=data.room_x[ind],
                room_y=data.room_y[ind],
            )
            data_list.append(data)

        return Actions(
            room_idx=torch.cat([data.room_idx for data in data_list], dim=0),
            room_x=torch.cat([data.room_x for data in data_list], dim=0),
            room_y=torch.cat([data.room_y for data in data_list], dim=0),
        )

    def sample(self, batch_size, episodes_per_file, hist_c) -> Actions:
        n = batch_size
        episodes_per_file = min(episodes_per_file, self.episodes_per_file)
        num_files = n // episodes_per_file

        t = torch.pow(torch.rand([num_files]), 1 / (1 + hist_c))
        file_num_list = torch.floor(t * self.num_files).to(torch.int64).clamp_max(self.num_files - 1).tolist()
        
        data = self.read_files(file_num_list, episodes_per_file)
        return data
