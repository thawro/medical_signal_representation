from torch.utils.data import Dataset

from data.ptxbl import load_ptbxl_split


class PTBXLDataset(Dataset):
    """PTB-XL Dataset class used in DeepLearning models."""

    def __init__(self, representation_type, fs, target, split, transform=None):
        dataset = load_ptbxl_split(representation_type, fs, target, split)
        self.data = dataset["data"]
        self.labels = dataset["labels"]
        self.classes = dataset["classes"]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx].float()
        data = torch.nan_to_num(data, nan=0.0)  # TODO !!!
        return data, self.labels[idx]
