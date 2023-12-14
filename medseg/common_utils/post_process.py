# Name: Tarun Saxena & Anson Antony
# CS 7180 Advanced Perception
# Date: 7 December, 2023

# post_process.py:- defines a function keep_largest_connected_components that retains only the largest connected components for each label in a segmentation mask using skimage.measure.

import numpy as np
from skimage import measure as measure


def keep_largest_connected_components(mask, n_classes):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''

    out_img = np.zeros(mask.shape, dtype=np.uint8)
    for struc_id in np.arange(1, n_classes):
        binary_img = mask == struc_id
        blobs = measure.label(binary_img, connectivity=1)
        props = measure.regionprops(blobs)

        if not props:
            continue
        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label
        out_img[blobs == largest_blob_label] = struc_id
    return out_img
