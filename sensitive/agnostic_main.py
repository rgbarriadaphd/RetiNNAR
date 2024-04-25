"""
# Author = ruben
# Date: 25/4/24
# Project: RetiNNAR
# File: agnostic_main.py

Description: main script to train retinal agnostic representation
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


import torch
from sensitive.triplet_dataset import EyePACSTripletDataset
from torch.utils.data import DataLoader
from utils import utils, logger

log = logger.Logger().log()
utils = utils.Utils("config/retinnar.json")
config = utils.get_config()


class AgnosticAgent:
    """Class to manage agnostic model train process"""

    def __init__(self):
        # Set device environment
        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Config device: {self._device}")

        # Force deterministic behaviour in random proces for the shake of reproducibility.
        torch.manual_seed(config["sensitive_model"]["seed"])

        # Learning variables
        self._triplet_dataset = None
        self._dataloader = None

    def _load_dataset(self):
        """Load json data and creates dataloader"""
        output_suffix = "_normalized" if config["triplets"]["normalize_embeddings"] else ""
        triplets_json = config["triplets"]["triplets_file"].format(suffix=output_suffix)
        self._triplet_dataset = EyePACSTripletDataset(triplets_json)

        # init dataloader
        self._dataloader = DataLoader(self._triplet_dataset,
                                      batch_size=config["sensitive_model"]["seed"],
                                      shuffle=True, num_workers=4, pin_memory=True)

    def _init_learning(self):
        """Initialize and define learning configuration"""
        self._load_dataset()

        for batch in self._dataloader:
            anchor, positive, negative, eq_label, dr_label = batch
            all(isinstance(x, torch.Tensor) for x in batch)

            print("*******")
            print(f"Anchor: {anchor.size()}")
            print(f"Positive: {positive.size()}")
            print(f"Negative: {negative.size()}")
            merged = torch.cat((anchor, positive, negative), dim=1)
            print(f"Merged: {merged.size()}")
            print(f"EQ Label: {eq_label}")
            print(f"DR Label: {dr_label}")
            print("*******")

            break

    def run(self):
        print("Run!")
        self._init_learning()


if __name__ == '__main__':
    aa = AgnosticAgent()
    aa.run()
