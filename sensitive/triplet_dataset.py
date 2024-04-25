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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import utils, logger

log = logger.Logger().log()
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
        self._triplet_dict = utils.load_json(file_path)

    def __len__(self):
        return len(self._triplet_dict)

    def __getitem__(self, item):
        """Retrieve item. Convert embeddings from list to tensors"""
        triplet = self._triplet_dict[str(item)]
        anchor_embedding = torch.tensor(triplet["anchor"]["embedding"])
        positive_embedding = torch.tensor(triplet["positive"]["embedding"])
        negative_embedding = torch.tensor(triplet["negative"]["embedding"])
        eq_label = torch.tensor(triplet["anchor"]["eq_label"])
        dr_label = torch.tensor(triplet["anchor"]["dr_label"])
        # print("========================")
        # print(f'({triplet["anchor"]["eq_label"]},{triplet["anchor"]["dr_label"]}) |'
        #       f' ({triplet["positive"]["eq_label"]},{triplet["positive"]["dr_label"]}) |'
        #       f' ({triplet["negative"]["eq_label"]},{triplet["negative"]["dr_label"]})')
        # print("========================")

        return anchor_embedding, positive_embedding, negative_embedding, eq_label, dr_label


if __name__ == '__main__':
    # Test triplet dataset. Load data
    output_suffix = "_normalized" if config["triplets"]["normalize_embeddings"] else ""
    triplets_json = config["triplets"]["triplets_file"].format(suffix=output_suffix)
    triplet_dataset = EyePACSTripletDataset(triplets_json)

    # init dataloader
    dataloader = DataLoader(triplet_dataset, batch_size=8,
                            shuffle=True, num_workers=4,
                            pin_memory=True)

    # iterate data
    for batch in dataloader:
        anchor, positive, negative, eq_label, dr_label = batch
        all(isinstance(x, torch.Tensor) for x in batch)

        log.info(f"Anchor: {anchor.size()}")
        log.info(f"Positive: {positive.size()}")
        log.info(f"Negative: {negative.size()}")
        merged = torch.cat((anchor, positive, negative))
        log.info(f"Merged: {merged.size()}")
        log.info(f"\n----\nEQ Label: {eq_label}")
        log.info(f"\n----\nDR Label: {dr_label}")

        break
