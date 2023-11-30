import glob
import os
import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image

from pycocotools import mask as mask_utils

import json
import numpy as np
from tqdm import tqdm


import torch.nn.functional as F






input_transforms = transforms.Compose([
    transforms.Resize((160, 256)),
    transforms.ToTensor(),
])

target_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((160, 256)),
])

class SA1B_Dataset(torchvision.datasets.ImageFolder):
    """A data loader for the SA-1B Dataset from "Segment Anything" (SAM)
    This class inherits from :class:`~torchvision.datasets.ImageFolder` so
    the same methods can be overridden to customize the dataset.
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.imgs[index] # discard automatic subfolder labels
        image = self.loader(path)
        masks = json.load(open(f'{path[:-3]}json'))['annotations'] # load json masks
        target = []

        for m in masks:
            # decode masks from COCO RLE format
            target.append(mask_utils.decode(m['segmentation']))
        target = np.stack(target, axis=-1)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target[target > 0] = 1 # convert to binary masks

        return image, target

    def __len__(self):
        return len(self.imgs)


input_reverse_transforms = transforms.Compose([
    transforms.ToPILImage(),
])

import matplotlib.pyplot as plt
def show_image(image, target, row=12, col=12):
    # image: numpy image
    # target: mask [N, H, W]
    fig, axs = plt.subplots(row, col, figsize=(20, 12))
    for i in range(row):
        for j in range(col):
            if i*row+j < target.shape[0]:
                axs[i, j].imshow(image)
                axs[i, j].imshow(target[i*row+j], alpha=0.5)
            else:
                axs[i, j].imshow(image)
            axs[i, j].axis('off')
    plt.tight_layout()
    plt.show()


from segment_anything import SamPredictor
import cv2



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 



# TODO: what if no 3 masks exist based on the points
def point_sample(img, target_mask):
    # TODO: negative point selection
    indices = torch.nonzero(target_mask > 0.1, as_tuple=True)
    random_idx = torch.randint(0, indices[0].size(0), (1,))

    # selected_point = (indices[0][random_idx], indices[1][random_idx])
    selected_point = torch.tensor([indices[0][random_idx], indices[1][random_idx]])

    selected_point = selected_point[None, None, :]
    return selected_point


# TODO: bbox
# def box_sample(img, target_mask):
#     return selected_bbox



def mask_focal_loss(prediction, targets):
    # TODO: focal loss
    alpha = 0.25
    gamma = 2

    loss = torch.nn.BCELoss()
    # loss = torch.nn.CrossEntropyLoss()
    focal_loss = loss(prediction, targets)

    return focal_loss.mean()


def mask_dice_loss(prediction, targets):
    
    inter = (prediction * targets).sum()
    union = (prediction * prediction).sum() + (targets * targets).sum()
    epsilon = 1e-7
    dice_loss = 1 - (2. * inter + epsilon) / (union + epsilon)

    return dice_loss

def iou_token_loss(iou_prediction, prediction, targets):
    num_masks = 1 # TODO: batch masks
    intersection = torch.sum(prediction * targets)
    union = torch.sum(prediction) + torch.sum(targets) - intersection
    epsilon = 1e-7
    iou_gt = intersection / (union + epsilon)

    iou_loss = F.mse_loss(iou_prediction, iou_gt, reduction='sum') / num_masks

    return iou_loss
    