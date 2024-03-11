from config import cfg

import os
import xml.etree.ElementTree as ET
import warnings

import numpy as np
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from helper import (shift_crop_training_sample, crop_sample,
                    Rescale, BoundingBox, cropPadImage, bgr2rgb)

warnings.filterwarnings("ignore")

class ImageNet_Dataset(Dataset):
    """ImageNet dataset class."""

    def __init__(self,
                 split : str, # train or val
                 path_to_img,
                 path_to_bb,
                 bb_params,
                 transform=None,
                 input_size=224):
        self.split = split
        self.path_to_img = path_to_img
        self.path_to_bb = path_to_bb
        self.transform = transform
        self.sz = input_size
        self.bb_params = bb_params
        self.x, self.y = self._parse_data(self.cfg.paths["imagenet"], self.cfg.paths["imagenetloc"])

    def __getitem__(self, idx):
        sample = self.get_sample(idx)
        if (self.transform):
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.len

    def _parse_data(self, image_dir, bbox_dir):
        print('Parsing ImageNet dataset...')
        classes = os.listdir(image_dir)
        x = []  # contains path to image files
        y = []  # contains bounding boxes
        x_dict = {}
        y_dict =self.get_bb(bbox_dir)
        for _class in classes:
            class_folder = os.path.join(image_dir, _class)
            for img in os.listdir(class_folder):
                x_dict[img] = os.path.join(class_folder, img)
    
        for img in x_dict:
            if img in y_dict:
                x.append(x_dict[img])
                y.append(y_dict[img])
        self.len = len(y)
        print('ImageNet dataset parsing done.')
        # should return 239283
        print('Total number of annotations in ImageNet dataset =', self.len)
        return x, y


    def get_bb(self, bbox_dir):
        bboxes = {}
        bbox_filepath = os.path.join(bbox_dir, "LOC_val_solution.csv")
        with open (bbox_filepath, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                # take the five first words
                img, annotation_line = line.split(",")
                classe = annotation_line.split(" ")[0]
                # bbox are 4 words, it can be more than 1 bbox per image
                # now we want to list all the bboxes with x_max, x_min, y_max, y_min
                bbox = []
                annotations_parsed = annotation_line.split(" ")[:-1]
                for i in range(0, len(annotations_parsed), 5):
                    bbox.append((annotations_parsed[i], (int(annotations_parsed[i+1]), int(
                        annotations_parsed[i+2]), int(annotations_parsed[i+3]), int(annotations_parsed[i+4]))))
                bboxes[img] = bbox 
        return bboxes

    def get_sample(self, idx):
        """
        Returns sample without transformation for visualization.

        Sample consists of resized previous and current frame with target
        which is passed to the network. Bounding box values are normalized
        between 0 and 1 with respect to the target frame and then scaled by
        factor of 10.
        """
        sample = self.get_orig_sample(idx)
        # unscaled current image crop with box
        curr_sample, opts_curr = shift_crop_training_sample(sample,
                                                            self.bb_params)
        # unscaled previous image crop with box
        prev_sample, opts_prev = crop_sample(sample)
        scale = Rescale((self.sz, self.sz))
        scaled_curr_obj = scale(curr_sample, opts_curr)
        scaled_prev_obj = scale(prev_sample, opts_prev)
        training_sample = {'previmg': scaled_prev_obj['image'],
                           'currimg': scaled_curr_obj['image'],
                           'currbb': scaled_curr_obj['bb']}
        return training_sample, opts_curr

    def get_orig_sample(self, idx):
        """
        Returns original image with bounding box at a specific index.
        Range of valid index: [0, self.len-1].
        """
        curr = cv2.imread(self.x[idx])
        curr = bgr2rgb(curr)
        currbb = self.y[idx]
        sample = {'image': curr, 'bb': currbb}
        return sample

    def filter_ann(self, ann):
        """
        Given list of ImageNet object annotations, filter objects which
        cover atleast 66% of the image in either dimension.
        """
        ans = []
        for an in ann:
            an_width = an[2]-an[0]
            an_height = an[3]-an[1]
            area_constraint = an_width > 0 and \
                an_height > 0 and an_width*an_height > 0
            if an_width <= (0.66)*self.sz[0] and \
               an_height <= (0.66)*self.sz[1] and \
               area_constraint:
                ans.append(an)
        return ans




if __name__ == "__main__":
    # test the dataset
    bb_params = {}
    bb_params['lambda_shift_frac'] = 5
    bb_params['lambda_scale_frac'] = 15
    bb_params['min_scale'] = -0.4
    bb_params['max_scale'] = 0.4
    imagenet = ImageNet_Dataset(
                                split='train',
                                path_to_img=cfg.paths["imagenet"],
                                path_to_bb=cfg.paths["imagenetloc"],
                                bb_params=bb_params,
                                transform=None,
                                input_size=224)
    print('Total number of samples in dataset =', len(imagenet))

