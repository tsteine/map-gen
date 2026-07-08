import torch

import os
import safetensors.torch
from env import Actions, EpisodeData


class ExperienceStorage:
    def __init__(self, num_rooms, data_path, episodes_per_file):
        self.num_rooms = num_rooms
        self.data_path = data_path
        self.episodes_per_file = episodes_per_file
        self.num_files = 0
        os.makedirs(data_path, exist_ok=True)

    def store(self, episode_data: EpisodeData):
        next_file_number = self.num_files
        assert episode_data.actions.room_idx.shape[0] == self.episodes_per_file
        assert episode_data.temperature.shape[0] == self.episodes_per_file
        assert episode_data.recommended_candidates.shape[0] == self.episodes_per_file
        assert episode_data.generation_variable_floats.shape[0] == self.episodes_per_file
        file_path = os.path.join(self.data_path, "{}.safetensors".format(next_file_number))
        safetensors.torch.save_file(
            {
                "room_idx": episode_data.actions.room_idx,
                "room_x": episode_data.actions.room_x,
                "room_y": episode_data.actions.room_y,
                "room_area": episode_data.actions.room_area,
                "temperature": episode_data.temperature,
                "recommended_candidates": episode_data.recommended_candidates,
                "generation_variable_floats": episode_data.generation_variable_floats,
            },
            file_path,
        )
        self.num_files += 1

    def read_files(self, file_num_list, episodes_per_file):
        data_list = []
        for file_num in file_num_list:
            file_path = os.path.join(self.data_path, "{}.safetensors".format(file_num))
            tensors = safetensors.torch.load_file(file_path)
            data = EpisodeData(
                actions=Actions(
                    room_idx=tensors["room_idx"],
                    room_x=tensors["room_x"],
                    room_y=tensors["room_y"],
                    room_area=tensors["room_area"],
                ),
                temperature=tensors["temperature"],
                recommended_candidates=tensors["recommended_candidates"],
                generation_variable_floats=tensors["generation_variable_floats"],
            )
            ind = torch.randperm(data.actions.room_idx.shape[0])[:episodes_per_file]
            data = EpisodeData(
                actions=Actions(
                    room_idx=data.actions.room_idx[ind],
                    room_x=data.actions.room_x[ind],
                    room_y=data.actions.room_y[ind],
                    room_area=data.actions.room_area[ind],
                ),
                temperature=data.temperature[ind],
                recommended_candidates=data.recommended_candidates[ind],
                generation_variable_floats=data.generation_variable_floats[ind],
            )
            data_list.append(data)

        return EpisodeData(
            actions=Actions(
                room_idx=torch.cat([data.actions.room_idx for data in data_list], dim=0),
                room_x=torch.cat([data.actions.room_x for data in data_list], dim=0),
                room_y=torch.cat([data.actions.room_y for data in data_list], dim=0),
                room_area=torch.cat([data.actions.room_area for data in data_list], dim=0),
            ),
            temperature=torch.cat([data.temperature for data in data_list], dim=0),
            recommended_candidates=torch.cat(
                [data.recommended_candidates for data in data_list], dim=0
            ),
            generation_variable_floats=torch.cat(
                [data.generation_variable_floats for data in data_list], dim=0
            ),
        )

    def sample(self, batch_size, episodes_per_file, hist_c) -> EpisodeData:
        n = batch_size
        episodes_per_file = min(episodes_per_file, self.episodes_per_file)
        num_files = (n + episodes_per_file - 1) // episodes_per_file

        t = torch.pow(torch.rand([num_files]), 1 / (1 + hist_c))
        file_num_list = (
            torch.floor(t * self.num_files).to(torch.int64).clamp_max(self.num_files - 1).tolist()
        )

        return self.read_files(file_num_list, episodes_per_file).slice(0, n)
