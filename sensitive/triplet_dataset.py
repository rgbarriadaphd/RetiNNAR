"""
# Author = ruben
# Date: 18/3/24
# Project: RetiNNAR
# File: triplet_dataset.py

Description: Class to implement triplet dataset extraction
"""
import torch
import os
import sys
from torch.utils.data import Dataset, DataLoader
from utils import utils

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

utils = utils.Utils("config/retinnar.json")
config = utils.get_config()


class EyePACSTripletDataset(Dataset):
    """
    Triplet dataset to manage triplet embeddings
    """
    def __init__(self, file_path: str):
        """
        Class constructor. Initialize dataset from triplet list
        :param file_path: (str) path to the json containing triplet data
        """
        self._triplet_list = utils.load_json(file_path)

    def __len__(self):
        return len(self._triplet_list)

    def __getitem__(self, item):
        """Retrieve item. Convert embeddings from list to tensors"""
        triplet = self._triplet_list[str(item)]
        anchor_embedding = torch.tensor(triplet["anchor"]["embedding"])
        anchor_label = torch.tensor(triplet["anchor"]["eq_label"])
        positive_embedding = torch.tensor(triplet["positive"]["embedding"])
        negative_embedding = torch.tensor(triplet["negative"]["embedding"])
        return anchor_embedding, positive_embedding, negative_embedding, anchor_label


if __name__ == '__main__':

    # Test triplet dataset. Load data
    triplet_dataset = EyePACSTripletDataset("eq_model/triplets.json")

    # init dataloader
    dataloader = DataLoader(triplet_dataset, batch_size=8,
                            shuffle=True, num_workers=4,
                            pin_memory=True)

    # iterate data
    for batch in dataloader:
        anchor, positive, negative, label = batch
        all(isinstance(x, torch.Tensor) for x in batch)
