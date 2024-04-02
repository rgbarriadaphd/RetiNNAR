"""
# Author = ruben
# Date: 18/3/24
# Project: RetiNNAR
# File: triplet_dataset.py

Description: Class to implement triplet dataset extraction
"""
import json
import os.path

import pandas as pd


class EyePACSTripletDataset:
    def __init__(self, dt_path, label_path, ):
        self._images = None

    def __len__(self):
        return len(self._images)

    def __getitem__(self, item):
        pass

        # if self.transform != None:
        #     anchor_img = self._transform(anchor_img)
        #     positive_img = self._transform(positive_img)
        #     negative_img = self._transform(negative_img)
        #
        # return anchor_img, positive_img, negative_img, anchor_label


if __name__ == '__main__':
    # Load EyePACS
    image_folder = "/home/ruben/PycharmProjects/RetiNNAR/input/train"
    fold_distribution = "/home/ruben/PycharmProjects/RetiNNAR/eq_model/eq_fold_distribution.json"

    csv_train = "/user/rgbarriada/shared/Eye-Quality/Label_EyeQ_train.csv"
    csv_test = "/user/rgbarriada/shared/Eye-Quality/Label_EyeQ_test.csv"
    df_test = pd.read_csv(csv_test)
    df_train = pd.read_csv(csv_train)

    data_classes = {"Good": 0, "Usable": 1, "Reject": 2}

    # read json
    input_json = "/user/rgbarriada/shared/RetiNNAR/eq_model/eq_fold_distribution.json"

    with open(input_json, 'r') as fp:
        data_dict = json.load(fp)


    convert_dataset_dict = {'train':[], 'test':[]}
    for fold_id in data_dict:
        for img_path, eq_label in data_dict[fold_id]:
            img_name = img_path.split('/')[-1]
            if 'train' in img_path:
                query = df_train[(df_train['image'] == img_name)]
            else:
                query = df_test[(df_test['image'] == img_name)]
            if len(query) > 0:
                # found
                query_dr_label = query['DR_grade']
                assert int(eq_label) == int(query['quality']), f'{int(eq_label)} != {int(query["quality"])}'
            else:
                # Not found
                raise NotImplemented

            stage = 'test' if fold_id == '1' else 'train'

            split_path = img_path.split('/')
            new_img_path = os.path.join("/".join(split_path[0:4]), 'Eye-Quality', "/".join(split_path[8:]))
            assert os.path.exists(new_img_path)

            convert_dataset_dict[stage].append((new_img_path, int(eq_label), int(query_dr_label)))

    # write json
    output_json = "/user/rgbarriada/shared/RetiNNAR/eq_model/image_distribution.json"
    with open(output_json, 'w') as fp:
        json.dump(convert_dataset_dict, fp)

    # # read json
    # input_json = "/home/ruben/PycharmProjects/RetiNNAR/eq_model/image_distribution.json"
    #
    # with open(input_json, 'r') as fp:
    #     data_dict = json.load(fp)
    #
    #
    #
    # print(data_dict)












