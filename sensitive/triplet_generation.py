"""
# Author = ruben
# Date: 20/3/24
# Project: RetiNNAR
# File: triplet_generation.py

Description: From image_distribution.json file, retrieve all possible triplets
"""
import random
import threading
import time
import os
import sys
import torch

import torch.nn as nn
from torchvision import models, transforms
from torch.nn.functional import normalize
from PIL import Image

from utils import utils

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

utils = utils.Utils("config/retinnar.json")
config = utils.get_config()


def get_eq_model() -> models.vgg16:
    """Retrieves VGG16 model trained for Eye Quality task"""
    vgg_model = models.vgg16()

    # Remove last layer
    num_features = vgg_model.classifier[6].in_features
    features = list(vgg_model.classifier.children())[:-1]

    # Substitution by the numer of classes in the last layer
    linear = nn.Linear(num_features, config["model"]["n_classes"])
    features.extend([linear])
    vgg_model.classifier = nn.Sequential(*features)

    # Load pretrained model
    eq_model_path = utils.build_path(config["dataset"]["eq"]["pretrained_model"])
    if torch.cuda.is_available():
        vgg_model.load_state_dict(torch.load(eq_model_path))
    else:
        vgg_model.load_state_dict(torch.load(eq_model_path, map_location=torch.device('cpu')))

    return vgg_model


def get_embedding(vgg_model: models.vgg16, image: str) -> torch.Tensor:
    """Returns the feature vector of corresponding layer"""

    # image transformation definition
    image_size = config["transforms"]["image_size"]
    norm_mean = config["transforms"]["normalization"]["mean"]
    norm_std = config["transforms"]["normalization"]["std"]
    data_transforms = transforms.Compose([
        transforms.Resize((image_size[0], image_size[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    # Retrieve activation vector of defined layer
    layer_idx, layer_name = config["model"]["activation_layer"]
    with open(image, "rb") as f:
        image = Image.open(f).convert("RGB")
        image = data_transforms(image)
        image = image.unsqueeze(0)
        vgg_model.classifier[layer_idx].register_forward_hook(get_activation(layer_name))
        vgg_model(image)
        return torch.flatten(activation[layer_name])


def satisfy_condition(x_anchor: torch.Tensor, x_positive: torch.Tensor, x_negative: torch.Tensor) -> bool:
    """
    Check condition to include triplet in the triplet list

    x_anchor, x_positive and x_negative are the feature vector of each image.
    anchor and positive are different images from the same class and negative is an image from a
    different class.

    The triplet is included in the list if satisfies:
        ‖ 𝐀 − 𝐍 ‖^2 − ‖ 𝐀 − 𝐏 ‖^2 < 𝛼
    where ‖∙‖ is the Euclidean Distance and 𝛼 is a real numbered threshold
    """
    a = torch.flatten(torch.cdist(x_anchor, x_negative, p=2)).item()
    b = torch.flatten(torch.cdist(x_anchor, x_positive, p=2)).item()
    threshold = config["triplets"]["triplet_threshold"]
    return (a - b) < threshold


activation = {}


def get_activation(name: str) -> torch.Tensor:
    """Retrieve activation from a specific layer of a model"""

    def hook(output):
        activation[name] = output.detach()

    return hook


class ComputeTripletsMultiThread:
    """Compute list of triplets from the original dataset. Uses multithreading"""

    def __init__(self, embeddings: dict, n_triplets: int):
        self._embeddings = embeddings
        self._triplet_list = []
        self._n_triplets = n_triplets

    def __call__(self) -> list:
        t_call = time.time()

        self._process_triplets()
        print(f"Processed embeddings: {len(self._embeddings)}")
        print(f"\n----\nTotal Elapsed time: {time.time() - t_call:.2f} s")
        print(f'List of triplets : {len(self._triplet_list)}')
        self._check_triplets()
        return self._triplet_list

    def save(self):
        # Convert to dict
        triplet_dict = {idx: e for idx, e in enumerate(self._triplet_list)}

        # Save triplets to file
        triplet_json = config["triplets"]["triplets_file"].format(suffix=output_suffix)
        utils.save_json(triplet_json, triplet_dict)

    def _check_triplets(self):
        """Triplet list verification

        In order to check if all triplets have been compound with different images for anchor, positive and negatives,
        in this method we create a list with the name of the triplet coded from the image file name from the 3 images:

        Example:
             for anchor:   '6820_right'
                 positive: '37010_right'
                 negative: '7775_left'

            The triplet name will be: '6820_right_37010_right_7775_left'

        Since order does not matter in this case we will have a triplet image path list such:

        [<triplet_name_1>, <triplet_name_2>, <triplet_name_3>,...,<triplet_name_N>] abd then the checks for duplicates
        is straightforward.
        """
        triplet_img_paths = [(f"{os.path.splitext(os.path.basename(t['anchor']['path']))[0]}_"
                              f"{os.path.splitext(os.path.basename(t['positive']['path']))[0]}_"
                              f"{os.path.splitext(os.path.basename(t['negative']['path']))[0]}")
                             for t in self._triplet_list]
        assert len(triplet_img_paths) == len(set(triplet_img_paths)), f"Triplet list contains duplicated items!"
        print("Triplet list verified correctly.")

    def _select_anchors(self) -> dict:
        anchor_selection = random.sample(list(self._embeddings), self._n_triplets)
        return {k: v for k, v in image_embeddings.items() if k in anchor_selection}

    def _process_triplets(self):

        # select anchors from original data
        anchors_dict = self._select_anchors()

        threads = []
        for anchor_id, anchor in enumerate(anchors_dict):
            # launch threads
            t = threading.Thread(target=self._inner_embedded_thread, args=((anchor_id, anchor),))
            threads.append(t)
            t.start()

            # Wait for all threads
        for t in threads:
            t.join()

    def _inner_embedded_thread(self, params):
        anchor_idx, anchor = params
        print(f"[{anchor_idx}] Thread start")
        t0 = time.time()
        # Obtain al positive and negatives from this anchor
        positive_list = self._retrieve_triplet_data(anchor, 'positive')
        negative_list = self._retrieve_triplet_data(anchor, 'negative')

        tries = 0
        found_triplet = False
        while not found_triplet:

            positive = random.sample(positive_list, k=1)[0]
            negative = random.sample(negative_list, k=1)[0]

            anchor_embedding = self._embeddings[anchor]["embedding"]
            positive_embedding = positive["embedding"]
            negative_embedding = negative["embedding"]

            if satisfy_condition(torch.tensor([anchor_embedding]), torch.tensor([positive_embedding]),
                                 torch.tensor([negative_embedding])):
                found_triplet = True
                # Fill triplet
                self._triplet_list.append({
                    "anchor": {
                        "path": self._embeddings[anchor]["path"],
                        "eq_label": self._embeddings[anchor]["eq_label"],
                        "dr_label": self._embeddings[anchor]["dr_label"],
                        "embedding": anchor_embedding
                    },
                    "positive": {
                        "path": positive["path"],
                        "eq_label": positive["eq_label"],
                        "dr_label": positive["dr_label"],
                        "embedding": positive_embedding
                    },
                    "negative": {
                        "path": negative["path"],
                        "eq_label": negative["eq_label"],
                        "dr_label": negative["dr_label"],
                        "embedding": negative_embedding
                    }
                })
            else:
                tries += 1
                print(f"[{anchor_idx}] ({len(self._embeddings)}, {len(positive_list)}, {len(negative_list)}) "
                      f"Not satisfactory triplet: {tries} attempts")

        print(f"[{anchor_idx}] Thread finish in {time.time() - t0:0.2f} s")

    def _retrieve_triplet_data(self, anchor_data: dict, triplet_type: str) -> list:
        """
        Returns the list of corresponding tuples of images, either positive or negative
        :param anchor_data: {'image_path' (str) :  eq_label (int), dr_label (int), embedding (list)}
        :param triplet_type: 'positive' | 'negative'
        :return: list of "triplet_type"
        """
        triplet_data = []

        anchor_path = anchor_data
        anchor_eq_label = self._embeddings[anchor_data]['eq_label']

        for candidate in self._embeddings:

            candidate_path = candidate
            candidate_eq_label = self._embeddings[candidate]['eq_label']

            if candidate_path == anchor_path:
                # same sample. Do not include
                continue

            triplet_info = self._embeddings[candidate]
            triplet_info.update({'path': candidate})

            if triplet_type == 'positive':
                if anchor_eq_label == candidate_eq_label:
                    triplet_data.append(triplet_info)

            elif triplet_type == 'negative':
                if anchor_eq_label != candidate_eq_label:
                    triplet_data.append(triplet_info)
            else:
                raise NotImplementedError
        return triplet_data


class ComputeEmbeddings:
    """Callable class to retrieve embeddings from images"""

    def __init__(self, dataset):
        self._image_dataset = dataset
        self._image_embeddings = {}

        # Get pretrained EQ model
        self._model = get_eq_model()

    def __call__(self):
        # For each anchor, obtain the feature vector from the corresponding activation layer
        for anchor in self._image_dataset:
            img_path = anchor[0]
            self._image_embeddings[img_path] = {'eq_label': anchor[1], 'dr_label': anchor[2]}
            embedding = get_embedding(self._model, img_path)

            # Normalize embedding
            if config["triplets"]["normalize_embeddings"]:
                embedding = normalize(embedding, dim=0)
            self._image_embeddings[img_path]['embedding'] = embedding.tolist()

    def save(self):
        # Save embeddings to file
        embeddings_suffix = "_normalized" if config["triplets"]["normalize_embeddings"] else ""
        embeddings_json = config["triplets"]["embeddings_file"].format(suffix=embeddings_suffix)
        utils.save_json(embeddings_json, self._image_embeddings)


if __name__ == '__main__':
    # Load train images distribution (this stage will only take into account fold 1)
    image_file = config["dataset"]["eq"]["image_distribution"]
    image_dataset = utils.load_json(image_file)['train']

    # From the original dataset, retrieve data and obtain feature vector (embeddings)
    compute_embeddings = ComputeEmbeddings(dataset=image_dataset)
    compute_embeddings()
    compute_embeddings.save()

    output_suffix = "_normalized" if config["triplets"]["normalize_embeddings"] else ""
    output_json = config["triplets"]["embeddings_file"].format(suffix=output_suffix)
    print(f'Loading image_embeddings file: {output_json}')
    t0 = time.time()
    image_embeddings = utils.load_json(output_json)
    print(f'[{time.time() - t0}] Successfully loaded {len(image_embeddings)} image_embeddings')

    # From the original dataset, retrieve data that forms a triplet
    compute_triplets = ComputeTripletsMultiThread(embeddings=image_embeddings, n_triplets=10000)
    compute_triplets()
    compute_triplets.save()
