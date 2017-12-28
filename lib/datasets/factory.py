# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.imagenet import imagenet
from datasets.our_dataset import our_dataset
import numpy as np

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012', '0712']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

for split in ['train', 'val']:
    name = 'imagenet_{}'.format(split)
    devkit_path = '/scratch0/ILSVRC/devkit/'
    data_path = '/scratch0/ILSVRC2015/'
    __sets[name] = (lambda split=split, devkit_path=devkit_path, data_path=data_path: imagenet(split, devkit_path, data_path))
    print name
    print __sets[name]

# for our dataset

__sets['our_dataset'] = (lambda image_set='our_dataset', data_path='/sdb/duyong/Datasets/objDetData/train/samples', img_list='../train_images_list.txt', label_list='../train_labels_list.txt' : our_dataset(image_set, data_path, img_list, label_list))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
