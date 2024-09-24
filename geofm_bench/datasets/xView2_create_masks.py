# xView2's tier3 subset does not come with image targets, so we need to create these first.
# Source: https://github.com/DIUx-xView/xView2_first_place/blob/master/create_masks.py
# Modification: 
# - Added root_dir for cases where the data is not in the current directory. 
# - Removed some unneeded imports and settings.
# - Pre-disaster masks take values {0,1} instead of {0,255}, for consistency with post-disaster masks.
"""
MIT License

Copyright (c) 2020 vdurnov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import json
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import pandas as pd
import cv2
import timeit
from os import path, makedirs, listdir
import sys
sys.setrecursionlimit(10000)
from multiprocessing import Pool

from shapely.wkt import loads
from shapely.geometry import mapping, Polygon


root_dir = "data/xView2/"
masks_dir = 'masks'

train_dirs = ['train', 'tier3', 'test']


def mask_for_polygon(poly, im_size=(1024, 1024)):
    img_mask = np.zeros(im_size, np.uint8)
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords)]
    interiors = [int_coords(pi.coords) for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


damage_dict = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 1 # ?
}


def process_image(json_file):
    js1 = json.load(open(json_file))
    js2 = json.load(open(json_file.replace('_pre_disaster', '_post_disaster')))

    msk = np.zeros((1024, 1024), dtype='uint8')
    msk_damage = np.zeros((1024, 1024), dtype='uint8')

    for feat in js1['features']['xy']:
        poly = loads(feat['wkt'])
        _msk = mask_for_polygon(poly)
        msk[_msk > 0] = 1

    for feat in js2['features']['xy']:
        poly = loads(feat['wkt'])
        subtype = feat['properties']['subtype']
        _msk = mask_for_polygon(poly)
        msk_damage[_msk > 0] = damage_dict[subtype]

    cv2.imwrite(json_file.replace('/labels/', '/masks/').replace('_pre_disaster.json', '_pre_disaster.png'), msk, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(json_file.replace('/labels/', '/masks/').replace('_pre_disaster.json', '_post_disaster.png'), msk_damage, [cv2.IMWRITE_PNG_COMPRESSION, 9])



if __name__ == '__main__':
    t0 = timeit.default_timer()

    all_files = []
    for d in train_dirs:
        makedirs(path.join(root_dir, d, masks_dir), exist_ok=True)
        for f in sorted(listdir(path.join(root_dir, d, 'images'))):
            if '_pre_disaster.png' in f:
                all_files.append(path.join(root_dir, d, 'labels', f.replace('_pre_disaster.png', '_pre_disaster.json')))


    with Pool() as pool:
        _ = pool.map(process_image, all_files)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))