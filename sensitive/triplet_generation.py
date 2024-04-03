"""
# Author = ruben
# Date: 20/3/24
# Project: RetiNNAR
# File: triplet_generation.py

Description: From image_distribution.json file, retrieve all possible triplets
"""
import os
import sys
import json
import torch
from tqdm import tqdm

import torch.nn as nn
from torchvision import models
from torchvision import transforms
from torch.nn.functional import normalize
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import utils
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


def retrieve_triplet_data(anchor_data: dict, images: dict, triplet_type: str) -> list:
    """
    Returns the list of corresponding tuples of images, either positive or negative
    :param anchor_data: {'image_path' (str) :  eq_label (int), dr_label (int), embedding (list)}
    :param images: dict with images definition {'image_path' (str) :  eq_label (int), dr_label (int), embedding (list)}
    :param triplet_type: 'positive' | 'negative'
    :return: list of "triplet_type"
    """
    triplet_data = []

    anchor_path = anchor_data
    anchor_eq_label = images[anchor_data]['eq_label']

    for candidate in images:

        candidate_path = candidate
        candidate_eq_label = images[candidate]['eq_label']

        if candidate_path == anchor_path:
            # same sample. Do not include
            continue

        if triplet_type == 'positive':
            if anchor_eq_label == candidate_eq_label:
                triplet_data.append(images[candidate])

        elif triplet_type == 'negative':
            if anchor_eq_label != candidate_eq_label:
                triplet_data.append(images[candidate])
        else:
            raise NotImplementedError
    return triplet_data


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


def satisfy_condition(vgg_model: models.vgg16, anchor_path: str, positive_path: str, negative_path: str) -> bool:
    """
    Check condition to include triplet in the triplet list

    anchor_path, positive_path, negative_path are strings of each image path.
    anchor and positive are different images from the same class and negative is an image from a
    different class.

    The triplet is included in the list if satisfies:
        ‖ 𝐀 − 𝐍 ‖^2 − ‖ 𝐀 − 𝐏 ‖^2 < 𝛼
    where ‖∙‖ is the Euclidean Distance and 𝛼 is a real numbered threshold
    """
    x_anchor = get_embedding(vgg_model, anchor_path)
    x_positive = get_embedding(vgg_model, positive_path)
    x_negative = get_embedding(vgg_model, negative_path)
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


if __name__ == '__main__':

    # Get pretrained EQ model
    model = get_eq_model()

    # Load train images distribution (this stage will only take into account fold 1)
    image_file = config["dataset"]["eq"]["image_distribution"]
    with open(image_file, 'r') as fp:
        image_dict = json.load(fp)
    train_image_data = image_dict['train']

    image_embeddings = {}
    with (tqdm(total=len(train_image_data), desc=f'Retrieve embeddings', unit='img') as pbar):
        # For each anchor, obtain the feature vector from the corresponding activation layer
        for anchor in train_image_data:
            img_path = anchor[0]
            image_embeddings[img_path] = {'eq_label': anchor[1], 'dr_label': anchor[2]}
            embedding = get_embedding(model, img_path)

            # Normalize embedding
            if config["triplets"]["normalize_embeddings"]:
                embedding = normalize(embedding, dim=0)

            image_embeddings[img_path]['embedding'] = embedding.tolist()
            pbar.update(1)

    # Save embeddings to file
    output_suffix = "_normalized" if config["triplets"]["normalize_embeddings"] else ""
    output_json = config["triplets"]["embeddings_file"].format(suffix=output_suffix)
    with open(output_json, 'w') as fp:
        json.dump(image_embeddings, fp)

    triplet_idx = 0
    triplet_list = {}
    with (tqdm(total=len(image_embeddings), desc=f'Triplet extraction process', unit='img') as pbar):
        for anchor in image_embeddings:
            # Obtain al positive and negatives from this anchor
            positive_list = retrieve_triplet_data(anchor, image_embeddings, 'positive')
            negative_list = retrieve_triplet_data(anchor, image_embeddings, 'negative')

            for p_id, positive in enumerate(positive_list):
                for n_id, negative in enumerate(negative_list):
                    image_positive = list(positive.keys())[0]
                    image_negative = list(negative.keys())[0]

                    if satisfy_condition(model, anchor, image_positive, image_negative):
                        # Fill triplet
                        triplet_list[triplet_idx] = {
                            "anchor": {
                                "path": image_embeddings[anchor],
                                "eq_label": image_embeddings[anchor]["eq_label"],
                                "dr_label": image_embeddings[anchor]["dr_label"],
                                "embedding": image_embeddings[anchor]["embedding"]
                            },
                            "positive": {
                                "path": image_embeddings[positive],
                                "eq_label": image_embeddings[positive]["eq_label"],
                                "dr_label": image_embeddings[positive]["dr_label"],
                                "embedding": image_embeddings[positive]["embedding"]
                            },
                            "negative": {
                                "path": image_embeddings[negative],
                                "eq_label": image_embeddings[negative]["eq_label"],
                                "dr_label": image_embeddings[negative]["dr_label"],
                                "embedding": image_embeddings[negative]["embedding"]
                            }
                        }
                        triplet_idx += 1
                    pbar.set_postfix(**{'P ': p_id})
            pbar.update(1)

    print(f'List of triplets : {len(triplet_list)}')
    # Save triplets to file
    output_json = config["triplets"]["triplets_file"].format(suffix=output_suffix)
    with open(output_json, 'w') as fp:
        json.dump(triplet_list, fp)
