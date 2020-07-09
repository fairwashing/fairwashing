import torch
import h5py

from torch.utils.data import Dataset


class ProjectedDataset(Dataset):
    def __init__(self, dataset, train=False, projection_file=None):
        super().__init__()

        self.dataset = dataset
        if projection_file:
            mode = 'train' if train else 'test'
            with h5py.File(projection_file, 'r') as fd:
                data = {key: fd['%s/%s' % (mode, key)][()] for key in ('v', 's')}

            self.projection = torch.from_numpy(data['v'])

    def _get_proj(self, idx):
        try:
            v = self.projection[idx]
        except AttributeError:
            return None
        else:
            return v

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        proj = self._get_proj(idx)
        if proj is None:
            return data, label
        else:
            return data, label, proj

    def __len__(self):
        return len(self.dataset)
