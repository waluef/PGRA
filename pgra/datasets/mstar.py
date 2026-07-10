"""Dataset loader."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _resize_numpy(img, target_size=(224, 224)):
    tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        tensor, size=target_size, mode="bilinear", align_corners=False)
    return resized.squeeze().numpy()


class MstarComponents(Dataset):
    def __init__(self, list_path, transform=None, part_name="ASC_part",
                 target_size=(224, 224)):
        self.transform = transform
        self.part_name = part_name
        self.target_size = target_size

        self._paths = []
        self._labels = []
        with open(list_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                self._paths.append(tokens[0])
                self._labels.append(int(tokens[1]))

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, idx):
        npz_path = self._paths[idx]
        data = np.load(npz_path)

        mag_img = np.abs(data["comp"]).squeeze()
        asc_part = np.abs(data[self.part_name])

        mag_img = _resize_numpy(mag_img, self.target_size)
        asc_part = np.stack([
            _resize_numpy(asc_part[..., c], self.target_size)
            for c in range(asc_part.shape[2])
        ], axis=2)

        if self.transform is not None:
            mag_img = self.transform(mag_img)
            asc_part = self.transform(asc_part)

        return mag_img, asc_part, self._labels[idx], npz_path
