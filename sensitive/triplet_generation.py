"""
# Author = ruben
# Date: 20/3/24
# Project: RetiNNAR
# File: triplet_generation.py

Description: From image_distribution.json file, retrieve all possible triplets
"""

import json
import torch
from tqdm import tqdm

import torch.nn as nn
from torchvision import models
from torchvision import transforms
from torch.nn.functional import normalize
from PIL import Image

from utils import utils
utils = utils.Utils("config/retinnar.json")
config = utils.get_config()

data_transforms = transforms.Compose([
            transforms.Resize((244, 244)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_eq_model():
    """Retrieves VGG16 model trained for Eye Quality task"""
    model = models.vgg16()
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]  # Remove last layer
    linear = nn.Linear(num_features, 3)
    features.extend([linear])
    model.classifier = nn.Sequential(*features)
    eq_model_path = config["dataset"]["eq"]["pretrained_model"]
    model.load_state_dict(torch.load(eq_model_path))

    return model


def retrieve_triplet_data(anchor_data: tuple, image_list: list, triplet_type: str) -> list:
    """
    Returns the list of corresponding tuples of images, either positive or negative
    :param anchor_data: (<img_path>: str,  eq_label: int, dr_label: int)
    :param image_list: list of tuples of images definition
    :param triplet_type: 'positive' | 'negative'
    :return: list of "triplet_type"
    """
    triplet_data = []
    label_target = 1

    for candidate in image_list:
        img_path = candidate[0]
        if img_path == anchor_data[0]:
            # same sample
            continue

        label = candidate[label_target]
        if triplet_type == 'positive':
            if label == anchor_data[label_target]:
                triplet_data.append(candidate)

        elif triplet_type == 'negative':
            if label != anchor_data[label_target]:
                triplet_data.append(candidate)
        else:
            raise NotImplementedError
    return triplet_data


def get_embedding(model, image):
    with open(image, "rb") as f:
        image = Image.open(f).convert("RGB")
        image = data_transforms(image)
        image = image.unsqueeze(0)
        model.classifier[3].register_forward_hook(get_activation('class_3'))
        model(image)
        return torch.flatten(activation['class_3'])


def satisfy_condition(model, image_anchor, image_positive, image_negative):
    x_anchor = get_embedding(model, image_anchor)
    x_positive = get_embedding(model, image_positive)
    x_negative = get_embedding(model, image_negative)
    a = torch.flatten(torch.cdist(x_anchor, x_negative, p=2)).item()
    b = torch.flatten(torch.cdist(x_anchor, x_positive, p=2)).item()
    return (a - b) < 0.2


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# def test_image(model, image_test):
#     # Load image
#     with open(image_test, "rb") as f:
#         image = Image.open(f).convert("RGB")
#         image = data_transforms(image)
#     image = image.unsqueeze(0)
#
#     model.classifier[3].register_forward_hook(get_activation('class_3'))
#     out = model(image)
#     emmbeding = activation['class_3']
#
#     print(emmbeding)


if __name__ == '__main__':

    model = get_eq_model()

    image_file = "/user/rgbarriada/shared/RetiNNAR/eq_model/image_distribution.json"

    with open(image_file, 'r') as fp:
        image_dict = json.load(fp)

    train_image_data = image_dict['train']



    totl = len(train_image_data)
    image_embeddings = {}
    with (tqdm(total=len(train_image_data), desc=f'Triplet extraction process', unit='img') as pbar):
        for anchor in train_image_data:
            img_path = anchor[0]
            image_embeddings[img_path] = {'eq_label': anchor[1], 'dr_label': anchor[2]}
            embedding = get_embedding(model, img_path)

            n_embedding = normalize(embedding, dim=0)
            image_embeddings[img_path]['embedding'] = n_embedding.tolist()
            pbar.update(1)

    output_json = "/user/rgbarriada/shared/RetiNNAR/eq_model/image_embeddings_norm.json"
    with open(output_json, 'w') as fp:
        json.dump(image_embeddings, fp)



    # cont = 0
    # triplet_list = []
    # with (tqdm(total=len(train_image_data), desc=f'Triplet extraction process', unit='img') as pbar):
    #     for anchor in train_image_data:
    #         positive_list = retrieve_triplet_data(anchor, train_image_data, 'positive')
    #         negative_list = retrieve_triplet_data(anchor, train_image_data, 'negative')
    #
    #         for p_id, positive in enumerate(positive_list):
    #             for n_id, negative in enumerate(negative_list):
    #                 image_anchor = anchor[0]
    #                 image_positive = positive[0]
    #                 image_negative = negative[0]
    #                 anchor_eq_label = anchor[1]
    #                 anchor_dr_label = anchor[2]
    #                 if satisfy_condition(model, image_anchor, image_positive, image_negative):
    #                     triplet_list.append((image_anchor, image_positive, image_negative, anchor_eq_label, anchor_dr_label))
    #                 pbar.set_postfix(**{'P ': p_id})
    #                 # pbar.set_postfix(**{})
    #         pbar.update(1)
    #
    #         if cont == 25:
    #             break
    #         else:
    #             cont += 1
    #
    # print(f'List of triplets : {len(triplet_list)}')
